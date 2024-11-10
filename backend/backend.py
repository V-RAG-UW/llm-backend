import uuid
from pyngrok import ngrok
from flask import Flask, request, jsonify
import av
import cv2
import torch
import numpy as np
import tempfile
from huggingface_hub import hf_hub_download
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
# Load the model in half-precision
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
    return whisper(video_path)

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
    
    return ret[0].split("ASSISTANT:")[1].strip()
def getMetaData(video):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name
        temp_video_file.write(video.read())  # Assuming 'video' is a file-like object
    metadata = {
        "transcription": getTranscription(temp_video_path),
        "description": getDescription(temp_video_path)
    }
    
    return metadata

@app.route('/process_video', methods=['POST'])
def process_video():
    print(f"I hate certain minorities: {request.files}")
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    metadata = getMetaData(video_file)
    return jsonify(metadata), 200

if __name__ == '__main__':
    app.run(port=5000)