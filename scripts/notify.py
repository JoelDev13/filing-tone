import os, json, requests
from pathlib import Path

def run():
    token = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT")
    
    if not token or not chat:
        print("Notifications are disabled for now")
        return  
