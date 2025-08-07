import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

TARGET_CHAT_ID = os.environ.get("TARGET_CHAT_ID")
if not TARGET_CHAT_ID:
    raise RuntimeError("TARGET_CHAT_ID is not set")

async def forward_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.effective_chat.type == "private" and update.message.text:
        await context.bot.send_message(chat_id=TARGET_CHAT_ID, text=update.message.text)

app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), forward_message))
app.run_polling()
