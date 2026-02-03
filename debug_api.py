import requests
import base64
import json
import os

# Path to the user's uploaded image
IMAGE_PATH = r"C:/Users/Lenovo/.gemini/antigravity/brain/7e9683dc-dd9b-4c21-bb36-c3ebc03e975f/uploaded_media_1770127877761.png"
API_URL = "http://localhost:5000/api/detect-image"

def debug_api():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at {IMAGE_PATH}")
        return

    print(f"Reading image from: {IMAGE_PATH}")
    with open(IMAGE_PATH, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    payload = {
        "image": image_data,
        "filename": "debug_test.png",
        "include_ela": True,
        "include_metadata": True,
        "include_watermark": True,
        "include_c2pa": True,
        "include_ai_analysis": True
    }

    print(f"Sending request to {API_URL}...")
    print(f"Payload size: {len(json.dumps(payload)) / 1024 / 1024:.2f} MB")

    try:
        response = requests.post(API_URL, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n--- Metadata Extraction Result ---")
            if 'metadata' in data:
                print(json.dumps(data['metadata'], indent=2))
            else:
                print("No 'metadata' key in response.")
            
            print("\n--- Full Response Summary ---")
            # Print keys to show what else is there
            print(f"Response keys: {list(data.keys())}")
        else:
            print("Error Response:")
            print(response.text)
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    debug_api()
