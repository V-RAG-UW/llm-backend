import base64
import asyncio
import aiohttp
import time
import os
from typing import List
from flask import Flask, request, jsonify
from llama_index.retrievers.videodb.base import NodeWithScore
from werkzeug.utils import secure_filename
import videodb
from videodb import SceneExtractionType, IndexType
from llama_index.retrievers.videodb import VideoDBRetriever

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Initialize Flask app
app = Flask(__name__)

# Temporary upload folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Establish connection to VideoDB
conn = videodb.connect(api_key=os.getenv("VIDEO_DB_API_KEY"))
retriever_spoken_words = VideoDBRetriever(api_key=os.getenv("VIDEO_DB_API_KEY"), index_type=IndexType.spoken_word, search_type="semantic", result_threshold=3, score_threshold=0.2)
retriever_scene = VideoDBRetriever(index_type=IndexType.scene, search_type="semantic", result_threshold=3, score_threshold=0.2)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        # Check if a file is part of the request
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        
        # Secure the filename and save it to the upload folder
        filename = secure_filename(video_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(file_path)
        
        # Upload the video to VideoDB
        video = conn.upload(file_path=file_path)
        
        # Index scenes in the video
        video.index_scenes(
            prompt="Describe the scene in strictly 100 words, do not add any system text in the description.", 
            extraction_type=SceneExtractionType.shot_based,
            extraction_config={
                "threshold": 20,
                "frame_count": 1
            },
        )
        video.index_spoken_words()
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return jsonify({"message": "Video uploaded and indexed successfully", "video_id": video.id}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

async def convert_frame_to_base64_async(frame_url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(frame_url) as response:
                # response.raise_for_status()
                # content = await response.read()
                # encoded_string = base64.b64encode(content).decode('utf-8')
                # return encoded_string
                return frame_url
    except Exception as e:
        return f"Error encoding frame: {str(e)}"

async def process_frames_async(frame_list: List[str]) -> List[str]:
    return await asyncio.gather(*(convert_frame_to_base64_async(frame.url) for frame in frame_list))

async def process_result_async(result: NodeWithScore, conn) -> dict:
    video = conn.get_collection().get_video(result.metadata["video_id"])
    # Print with expanded length
    title = result.node.metadata["title"]
    text = "" # result.text apparently returns the transcript if its a audio query. But we already do it at the bottom so no need!
    scene_collection = video.get_scene_collection(video.list_scene_collection()[0]["scene_collection_id"])
    scenes = scene_collection.scenes
    frames = []
    scene_counter = 0

    for scene in scenes:
        scene_counter += 1  # Increment scene counter
        # Append scene description with scene number and add a newline
        text += f" Scene {scene_counter}: {scene.description}\n"
        
        frame_list = scene.frames
        frame_base64_list = await process_frames_async(frame_list)
        frames.append({"scene_id": scene.id, "frames": frame_base64_list})

    return {
        "title": title,
        "text": text,
        "scenes": frames,
        "transcription": video.get_transcript_text()
    }

def union_nodes(list1: List[NodeWithScore], list2: List[NodeWithScore]) -> List[NodeWithScore]:
    # Create a dictionary to store nodes by their video_id
    node_dict = {}

    # Add nodes from the first list to the dictionary
    for node in list1:
        video_id = node.metadata["video_id"]
        if video_id not in node_dict:
            node_dict[video_id] = node

    # Add nodes from the second list, avoiding duplicates by video_id
    for node in list2:
        video_id = node.metadata["video_id"]
        if video_id not in node_dict:
            node_dict[video_id] = node

    # Return the unique nodes as a list
    return list(node_dict.values())

# Usage in the retrieval function
async def retrieval(visual_query: str, audio_query: str) -> List[dict]:
    # Run both retrievals asynchronously in separate threads
    spoken_words_task = asyncio.to_thread(retriever_spoken_words.retrieve, audio_query) if audio_query else []
    scene_task = asyncio.to_thread(retriever_scene.retrieve, visual_query) if visual_query else []

    # Await the tasks concurrently
    spoken_results, scene_results = await asyncio.gather(
        spoken_words_task if spoken_words_task else [],
        scene_task if scene_task else []
    )

    print("Length of spoken results:", len(spoken_results))
    print("Length of scene results:", len(scene_results))

    # Combine results using the custom union function
    combined_results = union_nodes(spoken_results, scene_results)

    # Process the combined results
    tasks = [process_result_async(result, conn) for result in combined_results]
    response = await asyncio.gather(*tasks)

    return response


@app.route('/rag_pipeline', methods=['POST'])
async def search_rag():
    try:
        visual_query = request.json.get('visual_query')
        audio_query = request.json.get('audio_query')
        
        if not visual_query and not audio_query:
            return jsonify({"error": "No search query provided"}), 400
        
        # Start benchmark timing
        start_time = time.time()
        
        result = await retrieval(audio_query=audio_query, visual_query=visual_query)
        
        # End benchmark timing
        end_time = time.time()
        total_time = end_time - start_time
        print("RAG pipeline benchmark:", total_time)
        return jsonify({"message": "Search completed", "result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port='6969')
