import os
import asyncio
from google import genai
from google.genai import types, errors

# 1. Setup API Key Rotation
# Expects a comma-separated string in Koyeb: key1,key2,key3
RAW_KEYS = os.getenv("GEMINI_KEYS", "")
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

if not API_KEYS:
    # Fallback to the single old variable if plural isn't found
    SINGLE_KEY = os.getenv("GEMINI_API_KEY")
    if SINGLE_KEY:
        API_KEYS = [SINGLE_KEY]

# Create a list of clients, one for each key
clients = [genai.Client(api_key=k) for k in API_KEYS]
current_key_index = 0

async def get_next_client():
    """Cycles through available API keys to distribute load."""
    global current_key_index
    if not clients:
        return None
    client = clients[current_key_index]
    current_key_index = (current_key_index + 1) % len(clients)
    return client

async def generate_response(user_id, prompt, history, image_data=None):
    """
    Handles text/multimodal requests using key rotation and 
    advanced error recovery.
    """
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.7
    )

    # Prepare parts
    new_message_parts = [{"text": prompt}]
    if image_data:
        new_message_parts.append(types.Part.from_bytes(
            data=image_data['bytes'], 
            mime_type=image_data['mime_type']
        ))
        
    contents = history + [{"role": "user", "parts": new_message_parts}]

    # --- RETRY & ROTATION LOGIC ---
    max_retries = len(API_KEYS) * 2 if API_KEYS else 3
    
    for attempt in range(max_retries):
        client = await get_next_client()
        if not client:
            return "❌ API Keys are missing. Please check environment variables."

        try:
            # Small delay to respect global Rate Limits
            await asyncio.sleep(1) 
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=config
            )
            return response.text
        
        except Exception as e:
            err_str = str(e).upper()
            if "429" in err_str or "QUOTA" in err_str or "EXHAUSTED" in err_str:
                print(f"Key {current_key_index} exhausted. Switching key...")
                # Wait briefly and try the next key in the next loop iteration
                await asyncio.sleep(2)
                continue
            else:
                print(f"Gemini Error: {e}")
                return f"❌ AI Error: {str(e)[:100]}..."

    return "⚠️ All API keys are currently rate-limited. Please try again in 1 minute."

async def generate_image(prompt):
    """Generates an image using available keys."""
    client = await get_next_client()
    if not client: return None

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
