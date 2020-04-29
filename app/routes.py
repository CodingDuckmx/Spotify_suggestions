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


@app.route('/run_model/<searched_song_id>', methods=['GET','POST'])
def song_recomendation(searched_song_id):

    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

               
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()

    searched_song_id = '2gZrHjEptqaEw7Hr40Gk8V'

    searched_song_name = sp.track(searched_song_id)['artists'][0]['name']
    searched_song_artist = sp.track(searched_song_id)['name']
    searched_song_features = {'acousticness':sp.audio_features(searched_song_id)[0]['acousticness'],
                            'danceability':sp.audio_features(searched_song_id)[0]['danceability'],
                            'energy':sp.audio_features(searched_song_id)[0]['energy'],
                            'instrumentalness':sp.audio_features(searched_song_id)[0]['instrumentalness'],
                            'key':sp.audio_features(searched_song_id)[0]['key'],
                            'liveness':sp.audio_features(searched_song_id)[0]['liveness'],
                            'loudness':sp.audio_features(searched_song_id)[0]['loudness'],
                            'mode':sp.audio_features(searched_song_id)[0]['mode'],
                            'speechiness':sp.audio_features(searched_song_id)[0]['speechiness'],
                            'tempo':sp.audio_features(searched_song_id)[0]['tempo'],
                            'time_signature':sp.audio_features(searched_song_id)[0]['time_signature'],
                            'valence':sp.audio_features(searched_song_id)[0]['valence'],
                            'popularity': sp.track(searched_song_id)['popularity']}  

    
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

    # prepare the info of the searched song
    searched_song_array = pd.Series(searched_song_features).values.reshape(1,songs_features.shape[1])

    # make the prediction
    prediction = knn.predict(searched_song_array)

    # Verify the searched song is not in the predictions.
    ten_most_similar_songs = songs.loc[prediction.argsort()[0][-11:]]
    if (ten_most_similar_songs['track_id'] == searched_song_id).any():
        ten_most_similar_songs = ten_most_similar_songs.drop(labels=ten_most_similar_songs['track_id'][ten_most_similar_songs['track_id'] == searched_song_id].index[0], axis=0)
    else:
        ten_most_similar_songs = ten_most_similar_songs[-10:]
    ten_most_similar_songs[['track_id','track_name','artist_name']]
    
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


@app.route('/song_search/<search>', methods=['GET','POST'])
def song_search(search):

    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


    results_dictionary_list = {}
    
    for i in range(len(sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'])):
        
        results_dictionary = {k:np.nan for k in ['artist', 'track_name', 'track_id']}
        results_dictionary['artist'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['artists'][0]['name']
        results_dictionary['track_name'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['name']
        results_dictionary['track_id'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['id']
        results_dictionary_list[i] = results_dictionary

    return json.dumps(results_dictionary_list)

@app.route('/search/<searched_artist>/<searched_song>', methods=['GET','POST'])
def complex_search(searched_artist,searched_song):

    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    results_dictionary_list = {}

    for i in range(len(sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'])):

        results_dictionary = {k:np.nan for k in ['artist', 'track_name', 'track_id']}
        results_dictionary['artist'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['artists'][0]['name']
        results_dictionary['track_name'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['name']
        results_dictionary['track_id'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['id']
        results_dictionary_list[i] = results_dictionary

    return json.dumps(results_dictionary_list)

@app.route('/artist_search/<search>', methods=['GET','POST'])
def artist_search(search):


    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


    results_dictionary = {}

    for i in range(len(sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'])):
    
        
        results_dictionary[i] = {'track_name': sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'][i]['name'], 'track_id': sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'][i]['id']}      
    
    return json.dumps(results_dictionary)