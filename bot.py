import os
import asyncio
import uvicorn
from fastapi import FastAPI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import ai_engine
import memory

app = FastAPI()

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "online"}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Safety Check: Ignore updates without messages
    if not update.message or not update.effective_user:
        return

    user_id = update.effective_user.id
    # Get text from message or caption (for photos)
    user_text = update.message.text or update.message.caption or ""
    
    # If it's a photo without text, still allow it to process
    if not user_text and not update.message.photo:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Get last 5 messages for context
        history = await memory.get_history(user_id, limit=5)
        
        # Check for images
        image_data = None
        if update.message.photo:
            photo = await update.message.photo[-1].get_file()
            image_bytes = await photo.download_as_bytearray()
            image_data = {'bytes': bytes(image_bytes), 'mime_type': 'image/jpeg'}

        # Generate AI Response
        response_text = await ai_engine.generate_response(user_id, user_text, history, image_data)
        
        # Save exchange to MongoDB
        await memory.save_message(user_id, "user", user_text)
        await memory.save_message(user_id, "model", response_text)
        
        await update.message.reply_text(response_text)
        
    except Exception as e:
        print(f"Error in handle_message: {e}")
        await update.message.reply_text("❌ Sorry, I encountered an error processing that.")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await memory.clear_history(user_id)
    await update.message.reply_text("🧹 Memory cleared! We're starting fresh.")

async def run_bot():
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Bot is online!")))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.ALL, handle_message))
    
    await application.initialize()
    await application.start()
    
    # FIX: drop_pending_updates=True prevents the Conflict error on Koyeb
    print("Starting polling...")
    await application.updater.start_polling(drop_pending_updates=True)

async def main():
    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    # Run FastAPI and Telegram Bot together
    await asyncio.gather(server.serve(), run_bot())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass        
