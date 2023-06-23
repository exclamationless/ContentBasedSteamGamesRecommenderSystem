import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("steam_games_exported.csv")
data.head()
data.info()

data_1 = data[["game_name", "popular_tags", "game_details", "game_description"]]
data_1.head(10)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix_multiple = tfidf.fit_transform(data_1["game_description"] + data_1["popular_tags"] + data_1["game_details"])

tfidf_matrix_multiple.toarray()

cosine_similarity_multiple = cosine_similarity(tfidf_matrix_multiple, tfidf_matrix_multiple)



def game_recommender_multiple(name, cosine_sim, data):
    indices = pd.Series(data.index, index=data['game_name'])
    indices = indices[~indices.index.duplicated(keep='last')]
    game_index = indices[name]
    similarity_scores = pd.DataFrame(cosine_sim[game_index], columns=["score"])
    game_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return data['game_name'].iloc[game_indices]


result=game_recommender_multiple("Stardew Valley", cosine_similarity_multiple, data_1)
print("-------")
print("Stardew Valley" + " Oyunu için öneriler:")
print(result)

