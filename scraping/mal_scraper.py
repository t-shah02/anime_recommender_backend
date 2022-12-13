import sys
sys.path.append("../")

import requests
from bs4 import BeautifulSoup as Soup
from db.crud import select_from_anime_table, insert_into_anime_table

BASE_MAL_URL = "https://myanimelist.net/anime/"
IMAGE_NOT_FOUND_DEFAULT_URL = "https://cdn.myanimelist.net/img/sp/icon/apple-touch-icon-256.png"

def get_anime_id_pictures(mal_ids):
    mal_id_picture_map = {}
    
    for mal_id in mal_ids:
        result = select_from_anime_table(mal_id)
        picture_url = None

        if result:
            picture_url = result[0]
        else:
            url = f"{BASE_MAL_URL}{mal_id}"
            response = requests.get(url)
            soup = Soup(response.content, "html.parser")
            meta_og_image_tag = soup.find("meta", {"property" : "og:image"}, content=True)
            picture_url = meta_og_image_tag["content"] if meta_og_image_tag else IMAGE_NOT_FOUND_DEFAULT_URL
            
            insert_into_anime_table(mal_id, picture_url)
        
        mal_id_picture_map[mal_id] = picture_url


    return mal_id_picture_map


