from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import base64
from mimetypes import guess_type
import time
import json
import threading
from PIL import Image
from io import BytesIO
import requests

from pyngrok import ngrok
from flask import Flask, request, jsonify

port = 4567
endpoint = ngrok.connect(port).public_url
print(endpoint)
app = Flask(__name__)

gpt = ChatOpenAI(model="gpt-4o-mini")
llama = ChatOllama(model="llama3.2", temperature=0)
stream = []

instruction = '''
Compare a single image of the user to reference images extracted from a video of the same user, focusing on:

Facial Features: Similarities or changes in appearance.
Expression and Pose: Differences in expression, gaze, or head position.
Lighting and Tone: Variations in lighting, shadows, or color.
Background: Any changes in surroundings.

Provide a summary of how closely the image matches the video frames, noting any major differences as an unformatted sentence.
'''
def get_comparison(reference_frames, frame):
    user_input = [
        {
            "type": "text",
            "text": "User Image:"
        },
        {
            "type": "image_url",
            "image_url": {"url": frame, "detail":"low"},
        },
        {
            "type": "text",
            "text": "The following frames are from the reference video:"
        },
    ]
    for ref_frame in reference_frames:
        user_input.append(
            {
                "type": "image_url",
                "image_url": {"url": ref_frame, "detail":"low"},
            },
        )
    prompt = ChatPromptTemplate.from_messages(
        [("system", instruction),
        ("user", user_input)]
    )
    chain = prompt | gpt | StrOutputParser()
    response = chain.invoke({})
    return response

def url_to_pil(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

def pil_to_base64(img, format="JPEG"):
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

def generate_context(metadata, reference_frame):
    context = []
    for result in metadata['result']:
        reference_frames = []
        for item in result['scenes']:
            for frame in item['frames']:
                reference_frames.append(pil_to_base64(url_to_pil(frame)))
        resp = get_comparison(reference_frames, reference_frame)
        context.append({"audio_transcription": result['transcription'], "difference_to_user": resp})
    return context

def handle_stream(question, reference_frame, desc, metadata):
    prompt = PromptTemplate(
    template="""
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|>
    You are a human friend of the user. You respond naturally and personally, as if you are having an ongoing, friendly conversation with someone you know well. 
    You have a memory repository to remember details from previous interactions with the user. Focus on the difference_to_user to point out any potential changes in user appearance. 
    These interactions from yesterday will be marked as CONTEXT.
    
    You also have access to previous dialogues in your current interactions, which will be marked as DIALOGUE, that includes the full ongoing conversation. 
    This stream includes everything the user has shared in this session, as well as your previous responses, allowing you to maintain a conversational flow that feels connected and engaged.
    Try not to ask or repeat information already establushed in this sections.
    
    Always engage positively, offering helpful, thoughtful, and supportive responses. Respond as though you are familiar with the user in a casual manner.
    OMIT SAYING things like you don't have personal experinces or memories.

    FOCUS ON THE USER INPUT QUESTION PRIMARILY AND REFER TO CONTEXT/DIALOGUES IF NECESSARY
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    #INPUT
    Scene Description: {desc}
    User Input: {question}
    
    #CONTEXT
    {context}
    
    #DIALOGUE
    {stream}
    
    #RESPONSE
    Answer length in sentence: {length}
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
    context = generate_context(metadata, reference_frame)
    rag_chain = prompt | llama | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "desc": {desc}, "stream": stream, "question": question, "length": 2})
    stream.append({"User input":question, "LLM response":generation})
    if len(stream>10):
        stream = stream[-10:]
    return generation

def play_sample_stream():
    user_input = [
        "Hello, how are you today? Notice any change with my getup?",
        "I'm thinking of making a website for mrbeast's feasibles. What do you think about it?",
        "How would my existing ideas come in handy here?",
        "Would my previous experiences contribute to this idea? What other programming language should I learn?",
        "Anyways, what do you think of what I'm wearing today? How does it compare to what I wore yesterday?",
        "Do you remember what I wore last time we talked?",
        "Alright I'll head out for today. Can you summarize the conversation we just had today?",
        "See you, have a good day!",
    ]
    reference_frame = pil_to_base64(Image.open('sample/face.png'))
    desc = "a person"
    with open("sample/log.txt", "r") as f:
        metadata = json.load(f)
    for question in user_input:
        answer = handle_stream(question, reference_frame, desc, metadata)
        print(f"Q: {question}")
        print(f"A: {answer}\n")

@app.route('/get_response', methods=['POST'])
def handle_http_request():
    input_json = request.json
    question = input_json['question'] # string
    reference_frame = input_json['qureference_frameestion'] # byte64 img
    desc = input_json['desc'] # string
    metadata = input_json['metadata'] # json
    response = handle_stream(question, reference_frame, desc, metadata)
    return response

if __name__=="__main__":
    app.run(port=5000)
    # play_sample_stream() # uncomment this to test run a static stream
