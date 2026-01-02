import os
from google import genai
from google.genai import types

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_response(user_id, prompt, history, image_data=None):
    """
    Core function to get AI response. 
    Handles: Text, Images (Multi-modal), and Web Grounding.
    """
    
    # Configure tools: Google Search Grounding is active!
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7
    )

    # Build contents list (History + New Prompt)
    contents = history
    
    new_message_parts = [{"text": prompt}]
    if image_data:
        # image_data should be a dict: {"mime_type": "image/jpeg", "data": base64_str}
        new_message_parts.append(types.Part.from_bytes(
            data=image_data['bytes'], 
            mime_type=image_data['mime_type']
        ))
        
    contents.append({"role": "user", "parts": new_message_parts})

    # Call Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash", # Best balance for free tier
        contents=contents,
        config=config
    )
    
    return response.text

async def generate_image(prompt):
    """Uses the 'Imagen' integration within Gemini to create visuals."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-image", 
        contents=[f"Generate an image based on: {prompt}"],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"])
    )
    
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data # Returns base64/bytes of the image
    return None
