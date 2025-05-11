# whisper.py
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

WHISPER_API = os.getenv('WHISPER_API')
WHISPER_VERSION = os.getenv('WHISPER_VERSION')
WHISPER_KEY = os.getenv('WHISPER_KEY')
WHISPER_NAME = os.getenv('WHISPER_NAME')


# Function to transcribe customer audio complaints using the Whisper model

# Load the audio file.
def read_audio_file(file_path):
    return open(file_path, "rb")

# Call the Whisper model to transcribe the audio file.
def whisper_openai_client(api_version, api_key, api_endpoint):
    client = AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )
    return client

# Extract the transcription and return it.
def stt(audio_file, whisper_deployment):
    result = whisper_deployment.audio.transcriptions.create(
        file=audio_file,
        model=WHISPER_NAME,
        response_format="text"
    )

    return result

def transcribe_audio():
    """
    Transcribes an audio file into text using OpenAI's Whisper model.

    Returns:
    str: The transcribed text of the audio file.
    """
    #Load the audio file.
    audio_segment = read_audio_file("audio/complain.m4a")
    
    #Call the Whisper model to transcribe the audio file.
    model_endpoint = whisper_openai_client(WHISPER_VERSION, WHISPER_KEY, WHISPER_API)

    #Extract the transcription and return it.
    text_result = stt(audio_segment, model_endpoint)
    
    with open("output/dtransciption.txt", "a") as f:
        f.write(text_result)
    
    return text_result

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    transcription = transcribe_audio()
    print(transcription)

# Example usage
