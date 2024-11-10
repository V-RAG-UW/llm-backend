# VideoDB powered RAG Server API

## Overview
This API enables users to upload video files, process them, and retrieve indexed data (scenes or spoken words) using a Retrieval-Augmented Generation (RAG) pipeline.

## Base URL
`http://<your-server-url>:6969`

## Endpoints

### 1. Upload Video
- **Endpoint**: `/upload_video`
- **Method**: `POST`
- **Description**: Uploads a video file, saves it temporarily, uploads it to VideoDB, indexes scenes, and spoken words, then removes the temporary file.
- **Request Body**:
  - **Type**: `multipart/form-data`
  - **Parameter**: `video` (The video file to upload)
- **Responses**:
  - **Success** (`201`):
    ```json
    {
      "message": "Video uploaded and indexed successfully",
      "video_id": "<video_id>"
    }
    ```
  - **Error** (`400`):
    ```json
    {
      "error": "No video file provided"
    }
    ```
  - **Error** (`500`):
    ```json
    {
      "error": "<error_message>"
    }
    ```

### 2. RAG Pipeline Search
- **Endpoint**: `/rag_pipeline`
- **Method**: `POST`
- **Description**: Processes a query using the RAG (Retrieval-Augmented Generation) pipeline to retrieve scenes or spoken words from the indexed video database and returns the processed results.
- **Request Body**:
  - **Type**: `application/json`
  - **Parameters**:
    - `query` (string, required): The search query for retrieval.
    - `type` (string, required): The type of search to perform (`"scene"` or `"spoken_word"`).
- **Responses**:
  - **Success** (`200`):
    ```json
    {
      "message": "Search completed",
      "result": [
        {
          "title": "<video_title>",
          "text": "<combined_text>",
          "scenes": [
            {
              "scene_id": "<scene_id>",
              "frames": ["<base64_encoded_frame1>", "<base64_encoded_frame2>", ...]
            }
          ],
          "transcription": "<audio_transcription of the video>",
        }
      ]
    }
    ```
  - **Error** (`400`):
    ```json
    {
      "error": "No search query provided"
    }
    ```
  - **Error** (`500`):
    ```json
    {
      "error": "<error_message>"
    }
    ```

## Example Requests

### Upload a Video
**cURL Command**:
```bash
curl -X POST http://<your-server-url>:6969/upload_video -F "video=@/path/to/your/video.mp4"
```

**Python Code**:
```python
import requests

url = 'http://<your-server-url>:6969/upload_video'
files = {'video': open('/path/to/your/video.mp4', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### RAG Pipeline Search
**cURL Command**:
```bash
curl -X POST http://<your-server-url>:6969/rag_pipeline -H "Content-Type: application/json" -d '{"query": "Find scene with character X", "type": "<scene or nothing>"}'
```

**Python Code**:
```python
import requests

url = 'http://<your-server-url>:6969/rag_pipeline'
data = {
    "query": "Find scene with character X",
    "type": "scene"
}
response = requests.post(url, json=data)
print(response.json())
```

## Environment Variables
Ensure that the following environment variables are set for the server:
- `VIDEO_DB_API_KEY`: API key for authenticating with VideoDB.

## Required Packages
Include the following in your `requirements.txt`:
```plaintext
flask>=2.3.0
aiohttp>=3.8.0
werkzeug>=2.3.0
python-dotenv>=0.19.0
videodb
llama_index 
llama-index-retrievers-videodb
```

## Running the Server
```bash
pip install -r "requirements.txt"
```
Run the server using:
```bash
python server.py
```

## Port Configuration
The server runs on port `6969`. You can modify the port by changing the `app.run(port='6969')` line in `server.py`.

---

Ensure to update the `<your-server-url>` placeholder with your actual server URL when testing the API.

