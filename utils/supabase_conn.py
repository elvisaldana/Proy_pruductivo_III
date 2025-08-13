# utils/supabase_conn.py
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

def conectar_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Faltan variables SUPABASE_URL o SUPABASE_KEY en el .env")
    return create_client(url, key)
