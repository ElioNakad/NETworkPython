import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

_db = None

def get_db():
    global _db
    if _db is None or not _db.is_connected():
        _db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", ""),
            port=int(os.getenv("DB_PORT", "3306")),
        )
    return _db
