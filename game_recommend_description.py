import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("steam_games_exported.csv")
data.head()
data.info()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_description = tfidf.fit_transform(data['game_description'])

cosine_similarity_description = cosine_similarity(tfidf_matrix_description, tfidf_matrix_description)

recommendations = {}

for idx, row in data.iterrows():
    similar_indices = cosine_similarity_description[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarity_description[idx][i], data['game_id'][i]) for i in similar_indices]

    recommendations[row['game_id']] = similar_items[1:]

def item(id):
    return data.loc[data['game_id'] == id]['game_name'].tolist()[0]


def game_recommender_description(item_id, num):
    print( str(num) + " adet " + item(item_id) + " oyununa benzer oyun öneriliyor...")
    recs = recommendations[item_id][:num]
    for rec in recs:
        print( item(rec[1]) + " - önerildi " + " (puan: " + str(rec[0]) + ")")


game_recommender_description(item_id=326, num=10)

