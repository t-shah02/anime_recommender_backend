from fastapi import FastAPI, Response
from nlp.predictions import get_predictions
import json

app = FastAPI()

@app.get("/genres")
def genres(response : Response, status_code=200):
    with open("./data/genres.json", "r") as file:
        genres = [genre.strip() for genre in json.loads(file.read())]
        return genres

@app.get("/predict")
async def predict(response : Response, score=None, genres=None, synopsis=None, num_recs=3, status_code=200):
    if score is None and genres is None and synopsis is None:
        response.status_code = 400
        return {"status" : "failure", "reason" : "Not enough query parameters to make any valid predictions"}

    predictions = get_predictions(score=score, genres=genres, synopsis=synopsis, num_recommendations=num_recs) 
    return {"status" : "success", "results" : predictions} 

