import psycopg2
import dotenv
import os

dotenv.load_dotenv()

CONN_STR = os.getenv("DATABASE_URL")



def create_anime_table():
    conn = psycopg2.connect(CONN_STR)
    cursor = conn.cursor()
    
    cursor.execute("CREATE TABLE ANIME \
                    (anime_id INTEGER PRIMARY KEY, \
                     picture_url VARCHAR (90))")
    
    conn.commit()
    conn.close()

def insert_into_anime_table(anime_id, picture_url):
    
    conn = psycopg2.connect(CONN_STR)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO ANIME \
                    (anime_id, picture_url) \
                    VALUES (%s, %s)", (anime_id, picture_url))
    
    conn.commit()
    conn.close()    



def select_from_anime_table(anime_id):
    conn = psycopg2.connect(CONN_STR)
    cursor = conn.cursor()

    cursor.execute("SELECT picture_url \
                            FROM ANIME \
                            WHERE anime_id = %s",(anime_id,))
    
    result = cursor.fetchone()

    conn.commit()
    conn.close() 
    
    return result
