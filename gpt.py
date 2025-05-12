from openai import AzureOpenAI
import json
import os

GPT_API = os.getenv('GPT_API')
GPT_VERSION = os.getenv('GPT_VERSION')
GPT_KEY = os.getenv('GPT_KEY')
GPT_NAME = os.getenv('GPT_NAME')


def classify_with_gpt(problem_list):
    """
    Classifies the customer complaint into a category/subcategory based on the image description.

    Returns:
    str: The category and subcategory of the complaint.
    """
    
    #Create a prompt that includes the image description and other relevant details.
    with open('categories.json', 'r') as file:
        classification_json = json.load(file)
    
    # Prompt to classify type of customer problem
    prompt = ("Classify the problem into classification list"
              "You need to identify (1) main class and (2) minor class"
              f"problem: {problem_list}"
              f"classification: {classification_json}"
        
    )

    #Call the GPT model to classify the complaint based on the prompt.
    gpt_client = AzureOpenAI(
        api_version=GPT_VERSION,
        api_key=GPT_KEY,
        azure_endpoint=GPT_API
    )

    #Extract and return the classification result.
    response = gpt_client.chat.completions.create(
        model=GPT_NAME,
        messages=[
            {"role": "system", "content": "You are a shop assistant to classify type of problem."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    classified_result = response.choices[0].message.content

    return classified_result
