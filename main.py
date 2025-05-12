# main.py

# Import functions from other modules
from whisper import transcribe_audio
from dalle import generate_image
from vision import describe_image, annotate_image, draw_bounding_box
from gpt import classify_with_gpt


def create_prompt(transcription):
    # write the transcription for extracting key details of customer complaint.
    return (
        "Generate a response to a customer complaint. "
        f"The customer says: '{transcription}'. "
        "Please provide a helpful response addressing their concerns."
    )

# Main function to orchestrate the workflow
def main(audio_file):
    """
    Orchestrates the workflow for handling customer complaints.
    
    Steps include:
    1. Transcribe the audio complaint.
    2. Create a prompt from the transcription.
    3. Generate an image representing the issue.
    4. Describe the generated image.
    5. Annotate the reported issue in the image.
    6. Classify the complaint into a category/subcategory pair.
    
    Returns:
    None
    """
    #1. Transcribe the audio complaint.
    stt_result = transcribe_audio(audio_file)
    print("Customer's complaint: ", stt_result)
    
    #2. Create a prompt from the transcription.
    customer_prompt = create_prompt(stt_result)

    #3. Generate an image representing the issue.
    generate_image(customer_prompt)
    
    #4. Describe the generated image.
    issue_list = describe_image(stt_result)
    print('Issue_list: ', issue_list)

    #5. Annotate the reported issue in the image. 
    coordinate_tuple = annotate_image(issue_list)
    draw_bounding_box(coordinate_tuple)
    
    #6. Classify the complaint into a category/subcategory pair.
    classification_result = classify_with_gpt(issue_list)

    #7. Print and store the results as required.
    print("Classification of problem: ", classification_result)
    with open("output/classification.txt", "a") as f:
        f.write(classification_result)
    
    return None

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    audio_file = "audio/complain_2.m4a"
    main(audio_file)
