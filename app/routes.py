import json
import numpy as np
import spotipy
import os
import pandas as pd
import psycopg2


from flask import current_app as app
from flask import render_template, request
from sklearn.preprocessing import LabelEncoder, scale
from spotipy.oauth2 import SpotifyClientCredentials
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


@app.route('/run_model/<searched_id>', methods=['GET','POST'])
def song_recomendation(searched_id):

    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

               
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()


    numerical_features = ['acousticness','danceability',
                                    'energy','instrumentalness','key', 'liveness',
                                    'loudness','mode','speechiness', 'tempo',
                                    'time_signature','valence','popularity']

    scaled_data = scale(songs[numerical_features])

    songs[numerical_features] = scaled_data


    features = ['acousticness','danceability',
                        'energy','instrumentalness','key', 'liveness',
                        'loudness','mode','speechiness', 'tempo',
                        'time_signature','valence','popularity']
    songs_features = songs[features].astype(float)

    #  Product of the vectors.

    knn = Sequential()
    knn.add(Dense(input_shape=(songs_features.shape[1],),
        units=songs.shape[0],
        activation='linear',
        use_bias=False)) 

    def normalize(vectors):

        norm_vectors = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norm_vectors)
    
    norm_songs_features = normalize(songs_features)


    # change the weights to the original matrix of features (of songs).
    knn.set_weights([np.array(norm_songs_features.T)])                    

    # random_choice = np.random.randint(1,songs.shape[0]+1)

    # songs[songs['track_id']==song_id].index[0]

    sample_song = songs.loc[songs[songs['track_id'] == searched_id ].index[0]]

    prediction = knn.predict(songs_features.loc[songs[songs['track_id']== searched_id ].index[0]].values.reshape(1,songs_features.shape[1]))

    ten_most_similar_songs = songs.loc[prediction.argsort()[0][-10:]]
    
    # 30 sec of song

    seconds_of_song_urls = []

    for i in range(10):
        seconds_of_song_urls.append(sp.tracks(ten_most_similar_songs['track_id'])['tracks'][i]['preview_url'])

    ten_most_similar_songs['sample_sound_url'] = seconds_of_song_urls

    artists_photos_urls = []

    for i in range(10):
        artists_photos_urls.append(sp.artist(sp.tracks(ten_most_similar_songs['track_id'])['tracks'][i]['artists'][0]['id'])['images'][0]['url'])

    ten_most_similar_songs['artists_photos_urls'] = artists_photos_urls

    result = json.dumps(ten_most_similar_songs.to_dict() )

    return result

