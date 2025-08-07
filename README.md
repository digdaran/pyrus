# Anonymous Telegram Group Bot

This bot forwards messages sent to it in a private chat to a target group while keeping the original sender anonymous.

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the API token.
2. Invite the bot to your target group and obtain the group's chat ID.
3. Set the following environment variables:

```
BOT_TOKEN=<telegram-bot-token>
TARGET_CHAT_ID=<target-group-id>
```

## Run

```
python bot.py
```

The bot listens for text messages sent to it in private and posts them to the specified group.
