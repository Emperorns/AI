import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "gemini_bot_db"
COLLECTION_NAME = "chat_history"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
history_collection = db[COLLECTION_NAME]

async def save_message(user_id, role, text):
    """Saves a message to MongoDB with a timestamp."""
    document = {
        "user_id": user_id,
        "role": role,
        "text": text,
        "timestamp": datetime.utcnow()
    }
    await history_collection.insert_one(document)

async def get_history(user_id, limit=5):
    """
    Retrieves the last N messages, optimized for Token usage.
    Automatically filters out non-text data to prevent 429 errors.
    """
    # Fetch latest messages sorted by newest first
    cursor = history_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
    messages = await cursor.to_list(length=limit)
    
    # Reverse to get chronological order (Oldest -> Newest)
    messages.reverse()
    
    formatted_history = []
    for msg in messages:
        # We only send text to the API for history. 
        # Sending old images/files is the main cause of 'Resource Exhausted'
        formatted_history.append({
            "role": msg["role"],
            "parts": [{"text": msg["text"]}]
        })
    
    return formatted_history

async def clear_history(user_id):
    """Deletes all history for a user (The /reset command)."""
    await history_collection.delete_many({"user_id": user_id})
    return True
