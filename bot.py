import os
import asyncio
import uvicorn
from datetime import datetime, timedelta
from fastapi import FastAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import ai_engine
import memory

# 1. FastAPI Setup (Updated for Koyeb Health Checks)
app = FastAPI()

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    """Responds to both GET and HEAD requests to pass Koyeb health checks."""
    return {"status": "online", "engine": "Gemini-2.0-Flash"}

# 2. Telegram Bot Logic
TOKEN = os.getenv("TELEGRAM_TOKEN")
user_cooldowns = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Advanced Gemini Bot Online!\n\nCommands:\n/draw [prompt] - Generate images\n/reset - Clear memory\nOr just send me text, photos, or voice!")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await memory.clear_history(update.effective_user.id)
    await update.message.reply_text("🧠 Memory cleared!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # Anti-Spam Cooldown (5 seconds)
    now = datetime.now()
    if user_id in user_cooldowns:
        if now < user_cooldowns[user_id] + timedelta(seconds=5):
            return # Ignore rapid messages to save quota
    user_cooldowns[user_id] = now

    user_text = update.message.text
    if not user_text: return

    # Let user know we are thinking
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Get history (limited to 5 messages to reduce token usage)
    history = await memory.get_history(user_id, limit=5)
    
    # Get AI Response
    response_text = await ai_engine.generate_response(user_id, user_text, history)
    
    # Save to MongoDB
    await memory.save_message(user_id, "user", user_text)
    await memory.save_message(user_id, "model", response_text)
    
    await update.message.reply_text(response_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("🎧 Listening to your voice message...")
    
    voice_file = await update.message.voice.get_file()
    voice_bytes = await voice_file.download_as_bytearray()
    
    history = await memory.get_history(user_id, limit=3)
    response = await ai_engine.generate_response(
        user_id, 
        "Transcribe and reply to this audio.", 
        history,
        image_data={"bytes": voice_bytes, "mime_type": "audio/ogg"}
    )
    await update.message.reply_text(response)

async def draw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Usage: /draw a futuristic city")
        return
    
    await update.message.reply_text("🎨 Generating image... please wait.")
    image_bytes = await ai_engine.generate_image(prompt)
    
    if image_bytes:
        await update.message.reply_photo(photo=image_bytes)
    else:
        await update.message.reply_text("❌ Sorry, image generation failed or quota reached.")

# 3. Execution Bridge
async def run_bot():
    application = Application.builder().token(TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("draw", draw))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.PHOTO, handle_message)) # Can expand this later
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Telegram Bot is Polling...")

async def main():
    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    
    # Run Web Server and Bot together
    await asyncio.gather(server.serve(), run_bot())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
