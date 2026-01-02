import os
import asyncio
from google import genai
from google.genai import types, errors

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_response(user_id, prompt, history, image_data=None):
    """
    Handles text and multi-modal requests with automatic retry on 429 errors.
    """
    # Google Search Grounding is active for real-time info
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7
    )

    contents = history
    new_message_parts = [{"text": prompt}]
    
    if image_data:
        new_message_parts.append(types.Part.from_bytes(
            data=image_data['bytes'], 
            mime_type=image_data['mime_type']
        ))
        
    contents.append({"role": "user", "parts": new_message_parts})

    # --- RETRY STRATEGY ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=config
            )
            return response.text
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = (attempt + 1) * 6  # Wait 6s, 12s, 18s
                print(f"Quota hit for user {user_id}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"Gemini Error: {error_msg}")
                return "❌ Sorry, I encountered an error processing that request."
    
    return "⚠️ The AI is currently at maximum capacity. Please try again in a few moments."

async def generate_image(prompt):
    """Generates an image using Gemini's native image capabilities."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[f"Generate an image of: {prompt}"],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
    except Exception as e:
        print(f"Image Gen Error: {e}")
    return None
