import uuid
import logging
from pyngrok import ngrok
from flask import Flask, request, jsonify
from flask_cors import CORS
import av
import cv2
import torch
import numpy as np
import tempfile
import requests
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
def read_video_cv2(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
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
    return ret[0].split("ASSISTANT:")[1].strip()
def getMetaData(video):
    logger.debug("About to read webm")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_webm_file:
        temp_webm_path = temp_webm_file.name
        temp_webm_file.write(video.read())
        logger.debug("Finished Reading webm")
    metadata = {
        "transcription": getTranscription(temp_webm_path),
        "description": getDescription(temp_webm_path)
    }
    
    return metadata

@app.route('/process_video', methods=['POST'])
def process_video():
    logger.debug("Obtained a request")
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files['video']
    metadata = getMetaData(video_file)
    rag_response = call_rag_pipeline(visual_query=metadata["description"], audio_query=metadata["transcription"])
    if isinstance(rag_response, tuple):
        return jsonify(rag_response[0]), rag_response[1]
    
    logger.debug(f"final data: {metadata}")
    return jsonify(metadata), 200

BASE_DB_URL = "http://127.0.0.1:6969"
def call_rag_pipeline(visual_query, audio_query):
    try:
        # Prepare the payload for the RAG pipeline API
        payload = {
            "visual_query": visual_query,
            "audio_query": audio_query
        }
        
        # Send POST request to the localhost RAG pipeline endpoint
        response = requests.post('http://localhost/rag_pipeline', json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
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

BASE_LLM_URL= "https://311f-72-33-2-197.ngrok-free.app/"
def generate_completions(question, reference_frame, scene_description, rag_metadata):
    try:
        # Prepare data for calling the RAG pipeline
        payload = {
            "question": question,
            "reference_frame": reference_frame,
            "desc": scene_description,
            "metadata": rag_metadata,
        }
        
        # Send a POST request to the /get_response endpoint
        response = requests.post(f'{BASE_LLM_URL}/get_response', json=payload)
        
        # Check the response status code
        if response.status_code == 200:
            return response.text  # Return the response as a string
        else:
            return f"Error: Received status code {response.status_code} with message: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(port=5000)