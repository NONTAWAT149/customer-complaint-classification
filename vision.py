# vision.py

from openai import AzureOpenAI
from mimetypes import guess_type
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import base64
import os
import ast

# Get API data
load_dotenv()
GPT_API = os.getenv('GPT_API')
GPT_VERSION = os.getenv('GPT_VERSION')
GPT_KEY = os.getenv('GPT_KEY')
GPT_NAME = os.getenv('GPT_NAME')


def local_image_to_data_url(image_path):
    # Identify path of image
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
        
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def openai_client(api_version, api_key, api_endpoint):
    # Call model from Azure API
    return AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )

def gpt_client(api_version, api_key, api_endpoint):
    # Call GPT model
    return openai_client(api_version, api_key, api_endpoint)

def describe_image(stt_result):
    """
    Describes an image and identifies key visual elements related to the customer complaint.

    Returns:
    str: A description of the image
    """
    
    # Load the generated image.
    data_url = local_image_to_data_url('output/generated_image.png')
    
    # Prompt to extract issue from customer complaint.
    prompt = (
            "From the image that you see, extract or identify problem of product."
            f"Here is also what customer say: {stt_result}"
            "List the problem or component in List of text."
            )
    
    # GPT model to identify customer's issue
    response = gpt_client(GPT_VERSION, GPT_KEY, GPT_API).chat.completions.create(
        model=GPT_NAME,
        messages=[
            {"role": "system", "content": "You are a shop assistant to help diagnose problem of product."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content
    

def annotate_image(issue_list):
    
    # Load image path
    data_url = local_image_to_data_url('output/generated_image.png')
    
    # Prompt to find bounding box of customer issue on image
    prompt = (
        "A task is to find bounding box of the problem on the image"
        "Identify the bounding box by identifing the centre location of bounding in coordinate (x, y)"
        f"Here is the list of problem: {issue_list}"
        "Return the output only coordinate data (x, y)"
    )
    
    # GPT model to identify location of issue
    response = gpt_client(GPT_VERSION, GPT_KEY, GPT_API).chat.completions.create(
        messages=[
                {"role": "system", "content": "You are a shop assistant to help point where the problem is on the image."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
                ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=GPT_NAME
    )
    coordinate_data = response.choices[0].message.content
    
    return ast.literal_eval(coordinate_data)


def draw_bounding_box(coordinate_tuple):
    
    # Load the image
    image_path = "output/generated_image.png"
    image = Image.open(image_path)

    # Create a Draw object
    draw = ImageDraw.Draw(image)

    # Define the text and position
    text = "Here is where the problem probably found."
    text_position = coordinate_tuple  # (x, y) coordinates

    # Draw the text on the image
    draw.text(text_position, text, fill="red")  # Use fill to set the text color

    # Draw a rectangle around the text
    rectangle_position = [text_position[0] - 10, text_position[1] - 10,
                        text_position[0] + 200, text_position[1] + 30]
    draw.rectangle(rectangle_position, outline="red", width=2)  # Draw a red rectangle

    # Save or display the annotated image
    image.save("output/annotated_image.png")  # Save the annotated image
    image.show()  # Display the annotated image
