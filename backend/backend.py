import uuid
import logging
from pyngrok import ngrok
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import av
import cv2
import torch
import numpy as np
import tempfile
import requests
import base64
import threading
import ffmpeg
import os
from huggingface_hub import hf_hub_download
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
# Load the model in half-precision

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map="cuda")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
whisper = pipeline("automatic-speech-recognition", "openai/whisper-tiny.en", torch_dtype=torch.float16, device="cuda:0")

# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
else:
    device = "cpu" 

print(f"Using device: {device}")
port = 5000
endpoint = ngrok.connect(port).public_url
print(endpoint)
app = Flask(__name__)
CORS(app)

def convert_to_mp4(video_path):
    try:
        # Check if the video is in webm format and convert to mp4
        if video_path.endswith('.webm'):
            mp4_path = video_path.replace('.webm', '.mp4')
            (
                ffmpeg
                .input(video_path)
                .output(mp4_path, vcodec='mpeg4', acodec='aac')
                .run(overwrite_output=True)
            )
            # os.remove(video_path)  # Remove the original webm file
            return mp4_path
        return video_path  # Return original path if not webm
    except Exception as e:
        logger.error(f"Error converting video: {e}")
        raise Exception("Failed to convert video to mp4")

def read_video_cv2(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    logger.error(f"total frames: {total_frames}")
    for frame_idx in indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.stack(frames)

def getTranscription(video_path):
    logger.debug("Generating transcription")
    trans = whisper(video_path)
    logger.debug(f"trans: {trans}")
    return trans

def getDescription(video_path):
    # Read 10 evenly spaced frames from video
    video = read_video_cv2(video_path, num_frames=10)
    key_frame = video[4]

    # Encode the key frame in base64
    _, buffer = cv2.imencode('.jpg', key_frame)
    key_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    conversation = [
        {
            "role": "system", 
            "content": [
                {"type": "text", "text": "You are given an video processed as a sequence to 15 images, Summarize the video, remembering to note the details and the key events that happen in the video (appearance, gesture, motion and more). Only output the summarization and nothing else."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=200)
    ret = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    logger.debug(f"disc: {ret[0].split("ASSISTANT:")[1].strip()}")
    return ret[0].split("ASSISTANT:")[1].strip(), key_frame_base64

def getMetaData(video):
    logger.debug("About to read webm")
    # Create a temporary file and ensure it's closed before further processing
    temp_webm_file = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    temp_webm_path = temp_webm_file.name
    temp_webm_file.write(video.read())
    temp_webm_file.close()    
    logger.debug("Finished Reading webm")
    
    # Convert webm to mp4
    mp4_path = convert_to_mp4(temp_webm_path)
    logger.debug("Done converting to mp4.")
    desc, key_frame_base64 = getDescription(mp4_path)
    metadata = {
        "transcription": getTranscription(temp_webm_path),
        "description": desc,
        'key_frame': key_frame_base64,
    }
    
    return metadata

def upload_video_non_blocking(video_path):
    def upload_video():
        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
                response = requests.post(f'{BASE_DB_URL}/upload_video', files=files)
                if response.status_code == 201:
                    logger.info("Video uploaded and indexed successfully")
                else:
                    logger.error(f"Error uploading video: {response.text}")
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
    
    threading.Thread(target=upload_video).start()

@app.route('/process_video', methods=['POST'])
def process_video():
    logger.debug("Obtained a request")
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files['video']
    video_path = f"/tmp/{video_file.filename}"
    # video_file.save(video_path)

    metadata = getMetaData(video_file)
    logger.debug(f"Input MetaData\n: {metadata}")
    rag_response = call_rag_pipeline(visual_query=metadata["description"], audio_query=metadata["transcription"])
    if isinstance(rag_response, tuple):
        return jsonify(rag_response[0]), rag_response[1]
    
    completions = generate_completions(question=metadata["transcription"], reference_frame=metadata["key_frame"], scene_description=metadata["description"], rag_metadata=rag_response)
    
    # Start non-blocking upload operation
    upload_video_non_blocking(video_path)

    # Stream the response from generate_completions
    return Response(completions, content_type='text/plain')


BASE_DB_URL = "http://0.0.0.0:6969"
def call_rag_pipeline(visual_query, audio_query):
    try:
        logger.debug(f"Piping to VRAG with visual query {visual_query}\n\n audio query {audio_query}")
        # Prepare the payload for the RAG pipeline API
        payload = {
            "visual_query": visual_query,
            "audio_query": audio_query
        }
        
        # Send POST request to the localhost RAG pipeline endpoint
        response = requests.post(f'{BASE_DB_URL}/rag_pipeline', json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            logger.debug(f"RAG error with {response.status_code}:{response.text}...\n")
            # Ensure the response is JSON or provide detailed feedback
            if 'application/json' in response.headers.get('Content-Type', ''):
                return {"error": response.json().get("error", "An unknown error occurred")}, response.status_code
            else:
                return {
                    "error": f"Unexpected response format: {response.text[:100]}",
                    "status_code": response.status_code
                }, response.status_code
        
    except requests.exceptions.JSONDecodeError:
        return {"error": "Failed to decode JSON from the response"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

BASE_LLM_URL= "https://6e9a-72-33-2-197.ngrok-free.app"
def generate_completions(question, reference_frame, scene_description, rag_metadata):
    try:
        # Prepare data for calling the RAG pipeline
        payload = {
            "question": question,
            "reference_frame": reference_frame,
            "desc": scene_description,
            "metadata": rag_metadata,
        }
        logger.debug(f"Generating Completions on query {question}...\n")
        
        # Send a POST request to the /get_response endpoint
        with requests.post(f'{BASE_LLM_URL}/get_response', json=payload, stream=True) as response:
            if response.status_code == 200:
                # Stream the response
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        logger.debug(chunk)
                        yield chunk.decode('utf-8')
            else:
                logger.debug(f"Completions error with {response.status_code}:{response.text}...\n")
                yield f"Error: Received status code {response.status_code} with message: {response.text}"
    except Exception as e:
        logger.debug(f"Completions error with {str(e)}.\n")
        yield f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(port=5000)