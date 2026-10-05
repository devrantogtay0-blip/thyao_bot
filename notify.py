"""notify.py - Opsiyonel Telegram bildirimi (TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID yoksa sessizce atlar)."""
import os


def send(text):
    token, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": str(text)[:3500]}, timeout=8)
        return bool(r.ok)
    except Exception:
        return False
