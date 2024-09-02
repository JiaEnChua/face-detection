import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv('API_URL')
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"}

def query(payload, retries=5, delay=2):
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.content
        else:
            return response.text
    return f"Error: Failed to generate image after {retries} attempts."

def generate_face_image_with_prompt(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "cartoon, illustration, anime, painting, drawing, unrealistic, low quality, blurry, deformed",
        }
    }

    result = query(payload)
    
    if isinstance(result, str):
        return result  # Return error message
    
    # If no error, process the image
    image = Image.open(io.BytesIO(result))
    return image

if __name__ == '__main__':
    prompt = "A photorealistic portrait of a taylor swift laughing, high quality, sharp focus, detailed facial features, natural lighting"
    result = generate_face_image_with_prompt(prompt)
    
    if isinstance(result, str):
        print(result)  # Print error message
    else:
        result.save("generated_image.png")
        print("Image has been generated and saved as 'generated_image.png'")