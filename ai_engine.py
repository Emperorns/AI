import os
import asyncio
from google import genai
from google.genai import types

# 1. API KEY ROTATION SETUP
# In Koyeb, set GEMINI_KEYS to a comma-separated list: key1,key2,key3
RAW_KEYS = os.getenv("GEMINI_KEYS", os.getenv("GEMINI_API_KEY", ""))
API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

# Initialize clients for all keys
clients = [genai.Client(api_key=k) for k in API_KEYS]
current_key_index = 0

async def get_next_client():
    global current_key_index
    if not clients:
        return None
    client = clients[current_key_index]
    current_key_index = (current_key_index + 1) % len(clients)
    return client

async def generate_response(user_id, prompt, history, image_data=None):
    """
    Generates a response using key rotation, history trimming, 
    and token optimization.
    """
    # 2. TOKEN OPTIMIZATION (THE MEMORY FIX)
    # We strip out heavy media from old messages. 
    # Only the current message should contain image data.
    optimized_history = []
    for msg in history:
        # Extract only text parts from previous turns
        text_parts = [p['text'] for p in msg.get('parts', []) if 'text' in p]
        if text_parts:
            optimized_history.append({
                "role": msg["role"],
                "parts": [{"text": " ".join(text_parts)}]
            })

    # 3. CONSTRUCT CURRENT MESSAGE
    current_parts = [{"text": prompt}]
    if image_data:
        current_parts.append(types.Part.from_bytes(
            data=image_data['bytes'], 
            mime_type=image_data['mime_type']
        ))
    
    optimized_history.append({"role": "user", "parts": current_parts})

    # 4. RETRY & ROTATION LOGIC
    # We try up to 3 different keys if we hit a rate limit
    max_attempts = min(len(clients), 3) if clients else 1
    
    for attempt in range(max_attempts):
        client = await get_next_client()
        if not client:
            return "❌ No API keys found in environment variables."

        try:
            # AI Configuration
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.7,
                system_instruction="You are a helpful AI assistant. Keep responses concise."
            )

            # Mandatory small delay to prevent rapid-fire hits on a single IP
            await asyncio.sleep(1)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=optimized_history,
                config=config
            )
            return response.text

        except Exception as e:
            error_msg = str(e).upper()
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"Key {current_key_index} rate limited. Retrying with next key...")
                continue # Try next key
            
            print(f"Gemini API Error: {e}")
            return f"⚠️ AI Error: {str(e)[:100]}..."

    return "🚀 System overloaded. Please try again in 30 seconds or use /reset."

async def generate_image(prompt):
    """Simple image generation fallback using the same key rotation."""
    client = await get_next_client()
    if not client: return None

    try:
        # Note: Image generation might require specific model settings in 2026
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[f"Generate an image of: {prompt}"],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
    except Exception:
        return None            
