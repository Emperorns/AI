import os
import asyncio
import uvicorn
from fastapi import FastAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import ai_engine
import memory

# 1. FastAPI Setup (For Koyeb Health Check)
app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "running", "bot": "Gemini Advanced"}

# 2. Telegram Bot Logic
TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! I'm your Gemini Advanced Bot. Send me text, images, or voice!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    
    # Get history from MongoDB
    history = await memory.get_history(user_id)
    
    # Get AI Response (with Web Grounding enabled in engine)
    response_text = await ai_engine.generate_response(user_id, user_text, history)
    
    # Save to Memory
    await memory.save_message(user_id, "user", user_text)
    await memory.save_message(user_id, "model", response_text)
    
    await update.message.reply_text(response_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    voice_file = await update.message.voice.get_file()
    voice_bytes = await voice_file.download_as_bytearray()
    
    # Gemini 2026 can process audio bytes directly
    response = await ai_engine.generate_response(
        user_id, 
        "Please transcribe and respond to this voice message.", 
        await memory.get_history(user_id),
        audio_data={"bytes": voice_bytes, "mime_type": "audio/ogg"}
    )
    await update.message.reply_text(f"🎙️ Voice recognized:\n\n{response}")

async def draw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt. Example: /draw a cyberpunk cat")
        return
    
    await update.message.reply_text("🎨 Generating your image...")
    image_bytes = await ai_engine.generate_image(prompt)
    if image_bytes:
        await update.message.reply_photo(photo=image_bytes)
    else:
        await update.message.reply_text("Failed to generate image.")

# 3. Execution Bridge
async def run_bot():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("draw", draw))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    print("Bot is polling...")

async def main():
    # Start both FastAPI and Telegram Bot
    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    
    await asyncio.gather(
        server.serve(),
        run_bot()
    )

if __name__ == "__main__":
    asyncio.run(main())
