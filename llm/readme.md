# LLM Chat Agent Backend

## Setup and Installation

### GPT-engine
To enable the GPT-powered vision model, set your OpenAI as an environment variable via:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### LLaMa completion engine
To enable LLaMa for chat completion, install ollama via the offical instructions https://ollama.com/

After ollama is installed, pull the latest version of LLaMa and host a server via
```bash
ollama pull llama3.2
ollama server
```
### Hosting
Run the following
```bash
python server.py
```
to start the server via flask-ngrok. A public endpoint will be available for further references.
