# whisper.py
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Download API data
load_dotenv()
WHISPER_API = os.getenv('WHISPER_API')
WHISPER_VERSION = os.getenv('WHISPER_VERSION')
WHISPER_KEY = os.getenv('WHISPER_KEY')
WHISPER_NAME = os.getenv('WHISPER_NAME')

def read_audio_file(file_path):
    # Load the audio file.
    return open(file_path, "rb")

def openai_client(api_version, api_key, api_endpoint):
    # Call model from Azure API
    return AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )

def stt(audio_file, client):
    # Extract the transcription and return it.
    return client.audio.transcriptions.create(
        file=audio_file,
        model=WHISPER_NAME,
        response_format="text"
    )

def transcribe_audio(audio_file):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.

    Returns:
    str: The transcribed text of the audio file.
    """
    
    # Load the audio file.
    audio_segment = read_audio_file(audio_file)
    
    # Call the Whisper model to transcribe the audio file.
    model_endpoint = openai_client(WHISPER_VERSION, WHISPER_KEY, WHISPER_API)

    # Extract the transcription and return it.
    text_result = stt(audio_segment, model_endpoint)
    
    # Save the transcription
    with open("output/transciption.txt", "a") as f:
        f.write(text_result)
    
    return text_result