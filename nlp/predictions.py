import pandas as pd 
import numpy as np 
import string
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import json

with open("./data/synonym_corpus.json","r") as file:
    word_synonym_corpus = json.loads(file.read())


df = pd.read_csv("./data/final_data.csv")
stopping_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
POS_TAG_MAP = {
    "N" : wordnet.NOUN,
    "V" : wordnet.VERB,
    "R" : wordnet.ADV,
    "J" : wordnet.ADJ
}



def normalize_synopsis(synopsis : str):
    synopsis = synopsis.lower()
    raw_tokens = word_tokenize(synopsis)
    clean_tokens = []

    for token in raw_tokens:
        
        # strip the token, in case it has leading or trailing spaces
        token = token.strip()

        # replace all english punctutation with an empty string, to improve the amount of tokens being real words
        for punct in string.punctuation:
            token = token.replace(punct, "")

        if token and token not in stopping_words:
            clean_tokens.append(token)

    final_tokens = set()
    for word, tag in pos_tag(clean_tokens):
        tag_type = POS_TAG_MAP.get(tag[0], "n")
        lemmatized_token = lemmatizer.lemmatize(word, tag_type)
        final_tokens.add(lemmatized_token)

    return " ".join(final_tokens)



def find_most_similar_anime_by_keywords(user_description, synonyms_corpus, target_df):  

    cleaned_user_description = normalize_synopsis(user_description)
    cleaned_user_tokens = set(cleaned_user_description.split(" "))

    def similarity_score(synopsis : str):
        synopsis_tokens = set(synopsis.split(" "))
        shared = cleaned_user_tokens.intersection(synopsis_tokens)
        user_not_shared = cleaned_user_tokens.difference(shared)
        synopsis_not_shared = synopsis_tokens.difference(shared)

        for w1 in user_not_shared:
            for w2 in synopsis_not_shared:
                w2_syns = synonyms_corpus.get(w2, [])
                if w1 in w2_syns:
                    shared.add(w1)
                    break
    
        return len(shared) / len(cleaned_user_tokens)



    target_df["similarity_score"] = target_df.normalized_synopsis.apply(similarity_score)

    return target_df


def jsonify_predictions(pred_df : pd.DataFrame):
    cols_to_keep = [col_name for col_name in pred_df.columns if col_name != "normalized_synopsis"]
    pred_df_json = pred_df[cols_to_keep].to_dict()   
    
    final_pred_json = []    
    

    mal_ids = pred_df_json["mal_id"].values()
    names = pred_df_json["name"].values()
    scores = pred_df_json["score"].values()
    synopses = pred_df_json["synopsis"].values()

    if "similarity_score" in pred_df_json:
        similarity_scores = pred_df_json["similarity_score"].values()

        for mal_id, name, score, synopsis, similarity_score in zip(mal_ids, names, scores, synopses, similarity_scores):
            pred_json = {
                "mal_id" : mal_id,
                "name" : name,
                "score" : score,
                "synopsis" : synopsis,
                "similarity_score" : similarity_score
            }

            final_pred_json.append(pred_json)

        return final_pred_json
    
    for mal_id, name, score, synopsis in zip(mal_ids, names, scores, synopses):
            pred_json = {
                "mal_id" : mal_id,
                "name" : name,
                "score" : score,
                "synopsis" : synopsis
            }

            final_pred_json.append(pred_json)

    return final_pred_json


def get_predictions(score=None, genres=None, synopsis=None, num_recommendations=3): 

    if score is None and genres is None and synopsis is None:
        return False

    pred_df = df.copy()

    SCORE_DIFF_THRESHOLD = 0.85

    if score:
        pred_df = pred_df[ abs(pred_df["score"] - np.float64(score)) <= SCORE_DIFF_THRESHOLD ]

    if genres:
        genres = genres.split(",")
        genre_bits_or = pred_df["genres"].str.contains(genres[0])

        for genre in genres[1:]:
            genre_bits_or |= pred_df["genres"].str.contains(genre)

        pred_df = pred_df[ genre_bits_or ]

    if synopsis:
        pred_df = find_most_similar_anime_by_keywords(synopsis, word_synonym_corpus, pred_df)  
        pred_df = pred_df.sort_values(by="similarity_score", ascending=False)

    pred_df = pred_df.head(n=num_recommendations)

    return jsonify_predictions(pred_df)