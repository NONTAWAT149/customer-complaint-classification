# dalle.py
# Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/dall-e-quickstart?tabs=command-line%2Ckeyless%2Ctypescript-keyless&pivots=programming-language-python 

from openai import AzureOpenAI
import requests
import os
from PIL import Image
import json

DALLE_API = os.getenv('DALLE_API')
DALLE_VERSION = os.getenv('DALLE_VERSION')
DALLE_KEY = os.getenv('DALLE_KEY')
DALLE_NAME = os.getenv('DALLE_NAME')


# Function to generate an image representing the customer complaint


def generate_image(prompt):
    """
    Generates an image based on a prompt using OpenAI's DALL-E model.

    Returns:
    str: The path to the generated image.
    """
    #Create a prompt to represent the customer complaint.
    client = AzureOpenAI(
        api_version=DALLE_VERSION,
        api_key=DALLE_KEY,
        azure_endpoint=DALLE_API
    )

    #Call the DALL-E model to generate an image based on the prompt.
    result = client.images.generate(model=DALLE_NAME, # the name of your DALL-E 3 deployment
                                    prompt=prompt,
                                    n=1)

    #Download the generated image and save it locally.
    json_response = json.loads(result.model_dump_json())

    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, 'output')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, 'generated_image.png')

    # Retrieve the generated image
    image_url = json_response["data"][0]["url"]  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)
    image.show()
