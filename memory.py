import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["gemini_bot_db"]
history_collection = db["chat_history"]

async def save_message(user_id, role, text):
    """Saves a message with a safety check on text content."""
    # Ensure we don't save 'None' values
    safe_text = text if text else "[Media or Empty Message]"
    document = {
        "user_id": user_id,
        "role": role,
        "text": safe_text,
        "timestamp": datetime.utcnow()
    }
    await history_collection.insert_one(document)

async def get_history(user_id, limit=5):
    """Retrieves history safely using .get() to avoid KeyError."""
    cursor = history_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    messages = await cursor.to_list(length=limit)
    messages.reverse() # Chronological order
    
    formatted_history = []
    for msg in messages:
        # FIX: msg.get("text", "") prevents 'KeyError: text'
        content = msg.get("text") or "[Media Content]"
        formatted_history.append({
            "role": msg["role"],
            "parts": [{"text": content}]
        })
    return formatted_history

async def clear_history(user_id):
    """Deletes history for the /reset command."""
    await history_collection.delete_many({"user_id": user_id})
