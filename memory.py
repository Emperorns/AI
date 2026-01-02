import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Get MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)
db = client["gemini_bot_db"]
memory_collection = db["chat_history"]

async def save_message(user_id, role, content):
    """Saves a message to MongoDB with a timestamp."""
    await memory_collection.insert_one({
        "user_id": user_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

async def get_history(user_id, limit=10):
    """Retrieves the last N messages for a specific user to provide context."""
    cursor = memory_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    history = await cursor.to_list(length=limit)
    # Reverse to keep chronological order for the AI
    return [{"role": h["role"], "parts": [{"text": h["content"]}]} for h in reversed(history)]

async def clear_history(user_id):
    """Wipes memory for a user if they use /reset."""
    await memory_collection.delete_many({"user_id": user_id})
