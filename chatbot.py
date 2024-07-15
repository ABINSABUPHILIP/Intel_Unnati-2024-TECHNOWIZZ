import os
import re
import string
import torch
import gradio as gr
from transformers import pipeline, LlamaTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Postprocessing function
def postprocess_response(response):
    response = response.strip()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    sentences = [sentence.capitalize() for sentence in sentences]
    response = ' '.join(sentences)
    return response

try:
    # Load the tokenizer and quantized model from the saved directory
    model_path = '/content/drive/MyDrive/openvino/chatbot/FINAL'
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    

    logger.info(f"Loading model from {model_path}")
    pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise e

# Function to generate response
def generate_response(user_input):
    try:
        user_input = preprocess_text(user_input)
        logger.info(f"Generating response for input: {user_input}")
        response = pipe(user_input, max_length=100, temperature=0.7, top_k=50, top_p=0.9, num_return_sequences=1)[0]['generated_text']
        response = postprocess_response(response)
        logger.info(f"Response generated: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response"

# Sample text generation prompts
sample_prompts = [
    "Write an e-mail",
    "Advantages of tinyllama",
    "Write a poem",
    "Write a code for :",
    "Best movies of all time ?"
]

# Define Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your queries "),
    outputs=gr.Textbox(label="Reply"),
    title="Chatbot using TinyLlama and Intel OpenVINO ",
    description="Welcome to Chatbot! Enter your questions ",
    examples=sample_prompts,
    theme="default",
    allow_flagging="never"
)

# Launch the Gradio interface
try:
    logger.info("Launching Gradio interface...")
    iface.launch(share=True)
    logger.info("Gradio interface launched successfully")
except Exception as e:
    logger.error(f"Error launching Gradio interface: {e}")