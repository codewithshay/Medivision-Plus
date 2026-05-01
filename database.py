import sqlite3
import pandas as pd

def create_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    # Table for user credentials
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (phone TEXT PRIMARY KEY, password TEXT, name TEXT, age INTEGER)''')
    # Table for diagnostic history
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (phone TEXT, organ TEXT, result TEXT, confidence REAL, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_user(phone, password, name, age):
    try:
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?,?,?)", (phone, password, name, age))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(phone, password):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE phone=? AND password=?", (phone, password))
    data = c.fetchone()
    conn.close()
    return data

def save_history(phone, organ, result, confidence):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO history (phone, organ, result, confidence) VALUES (?,?,?,?)", 
              (phone, organ, result, confidence))
    conn.commit()
    conn.close()

def get_history(phone):
    conn = sqlite3.connect('patients.db')
    df = pd.read_sql_query(f"SELECT organ, result, confidence, date FROM history WHERE phone='{phone}'", conn)
    conn.close()
    return df