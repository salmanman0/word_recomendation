from flask import Flask ,redirect,url_for,request,jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

@app.route('/')
def main():
    return 'Error!'

@app.route('/recommend/<keyword>')
def recoomend(keyword):
    word = keyword
    df = pd.read_csv('combined_dataset.csv')
    rows_with_none_values = df.applymap(lambda x: x == 'none').any(axis=1)
    df_cleaned = df[~rows_with_none_values]
    df_cleaned.head()
    df_cleaned.to_csv('df_cleaned.csv', index=False)
    # Membuat vektor fitur dari genre menggunakan CountVectorizer
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(df_cleaned['kategori'])
    # Menghitung kemiripan cosine antar word
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    # Fungsi untuk memberikan rekomendasi berdasarkan word
    def get_recommendations(word, num_recommendations=10):
        idx = df_cleaned[df_cleaned['kata'] == word].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [i for i in sim_scores if i[1] > 0 and i[0] != idx]
        word_indices = [i[0] for i in sim_scores]
        # Memilih secara acak dari daftar rekomendasi
        random_recommendations = random.sample(word_indices, min(num_recommendations, len(word_indices)))
        return df_cleaned['kata'].iloc[random_recommendations]
    
    # get rekomendasi
    word_recommendations = get_recommendations(word)
    _word_ = str(word_recommendations)
    _word_ = _word_.split("\n")
    for i in range(10) :
        new_word = _word_[i].split(" ")
        _word_[i] = new_word[-1]
    return jsonify({'result' : _word_})

if __name__ =='__main__':
    app.run('0.0.0.0', port=5000 , debug=True)