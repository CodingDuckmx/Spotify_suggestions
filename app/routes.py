import base64
import chart_studio.plotly as py
import io
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import spotipy
import os
import pandas as pd
import plotly.graph_objs as go
import psycopg2

from flask import current_app as app
from flask import jsonify
from flask import render_template, request
from plotly.subplots import make_subplots
# from sklearn.preprocessing import LabelEncoder, scale
from spotipy.oauth2 import SpotifyClientCredentials
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential


matplotlib.use('Agg')

client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


@app.route('/run_model/<searched_song_id>', methods=['GET','POST'])
def song_recomendation(searched_song_id):

         
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()

    

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
    if (ten_most_similar_songs['id'] == searched_song_id).any():
        ten_most_similar_songs = ten_most_similar_songs.drop(labels=ten_most_similar_songs['id'][ten_most_similar_songs['id'] == searched_song_id].index[0], axis=0)
    else:
        ten_most_similar_songs = ten_most_similar_songs[-10:]
    ten_most_similar_songs[['id','track_name','artist_name']]
    
    # 30 sec of song

    seconds_of_song_urls = []

    for i in range(10):
        seconds_of_song_urls.append(sp.tracks(ten_most_similar_songs['id'])['tracks'][i]['preview_url'])

    ten_most_similar_songs['sample_sound_url'] = seconds_of_song_urls

    artists_photos_urls = []

    for i in range(10):
        artists_photos_urls.append(sp.artist(sp.tracks(ten_most_similar_songs['id'])['tracks'][i]['artists'][0]['id'])['images'][0]['url'])

    ten_most_similar_songs['artists_photos_urls'] = artists_photos_urls

    result = json.dumps(ten_most_similar_songs.to_dict() )

    return result


@app.route('/song_search/<search>', methods=['GET','POST'])
def song_search(search):

    results_dictionary_list = {}
    
    for i in range(len(sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'])):
        
        results_dictionary = {k:np.nan for k in ['artist', 'track_name', 'id']}
        results_dictionary['artist'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['artists'][0]['name']
        results_dictionary['track_name'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['name']
        results_dictionary['id'] = sp.search(search, limit=10, offset=0, type='track', market=None)['tracks']['items'][i]['id']
        results_dictionary_list[i] = results_dictionary

    return json.dumps(results_dictionary_list)

@app.route('/search/<searched_artist>/<searched_song>', methods=['GET','POST'])
def complex_search(searched_artist,searched_song):

    results_dictionary_list = {}

    for i in range(len(sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'])):

        results_dictionary = {k:np.nan for k in ['artist', 'track_name', 'id']}
        results_dictionary['artist'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['artists'][0]['name']
        results_dictionary['track_name'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['name']
        results_dictionary['id'] = sp.search(q=f'artist:{searched_artist} track:{searched_song}')['tracks']['items'][i]['id']
        results_dictionary_list[i] = results_dictionary

    return json.dumps(results_dictionary_list)

@app.route('/artist_search/<search>', methods=['GET','POST'])
def artist_search(search):

    results_dictionary = {}

    for i in range(len(sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'])):
    
        
        results_dictionary[i] = {'track_name': sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'][i]['name'], 'id': sp.artist_top_tracks(sp.search(search, limit=10, offset=0, type='artist', market=None)['artists']['items'][0]['id'])['tracks'][i]['id']}      
    
    return json.dumps(results_dictionary)


@app.route('/plot/<searched_song_id>')

def build_plot(searched_song_id):

    img = io.BytesIO()
             
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()

   
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
    searched_song_array = (pd.Series(searched_song_features).values / np.linalg.norm(pd.Series(searched_song_features).values)).reshape(1,songs_features.shape[1])

    # make the prediction
    prediction = knn.predict(searched_song_array)

    # Verify the searched song is not in the predictions.
    ten_most_similar_songs = songs.loc[prediction.argsort()[0][-11:]]
    if (ten_most_similar_songs['id'] == searched_song_id).any():
        ten_most_similar_songs = ten_most_similar_songs.drop(labels=ten_most_similar_songs['id'][ten_most_similar_songs['id'] == searched_song_id].index[0], axis=0)
    else:
        ten_most_similar_songs = ten_most_similar_songs[-10:]
    ten_most_similar_songs[['id','track_name','artist_name']]
    
    extendend_result = ten_most_similar_songs.append(searched_song_features,ignore_index=True)
    extendend_result['outcome'] = 1
    extendend_result.iloc[10,-1] = 0
    extendend_result = extendend_result[['acousticness','danceability','energy','instrumentalness',
                        'key', 'liveness', 'loudness','mode','speechiness', 'tempo',
                        'time_signature','valence','outcome']]

    extendend_result_for_plot = extendend_result.T.reset_index()
    extendend_result_for_plot = extendend_result_for_plot.rename({0: 'Recomendation 1',1: 'Recomendation 2',2: 'Recomendation 3',
                                                                3: 'Recomendation 4',4: 'Recomendation 5',5: 'Recomendation 6',
                                                                6: 'Recomendation 7',7: 'Recomendation 8',8: 'Recomendation 9',
                                                                9: 'Recomendation 10',10: 'Base Song'}, axis=1)

    scatter0 = sns.scatterplot(x='index', y='Recomendation 1', data =extendend_result_for_plot)
    scatter1 = sns.scatterplot(x='index', y='Recomendation 2', data =extendend_result_for_plot)
    scatter2 = sns.scatterplot(x='index', y='Recomendation 3', data =extendend_result_for_plot)
    scatter3 = sns.scatterplot(x='index', y='Recomendation 4', data =extendend_result_for_plot)
    scatter4 = sns.scatterplot(x='index', y='Recomendation 5', data =extendend_result_for_plot)
    scatter5 = sns.scatterplot(x='index', y='Recomendation 6', data =extendend_result_for_plot)
    scatter6 = sns.scatterplot(x='index', y='Recomendation 7', data =extendend_result_for_plot)
    scatter7 = sns.scatterplot(x='index', y='Recomendation 8', data =extendend_result_for_plot)
    scatter8 = sns.scatterplot(x='index', y='Recomendation 9', data =extendend_result_for_plot)
    scatter9 = sns.scatterplot(x='index', y='Recomendation 10', data =extendend_result_for_plot)
    scatter10 = sns.scatterplot(x='index', y='Base Song', data =extendend_result_for_plot)

    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'  
    )




    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)

@app.route('/plotly/<searched_song_id>')

def build_plotly(searched_song_id):

    img = io.BytesIO()

    client_credentials_manager = SpotifyClientCredentials(client_id='64c7e99146a749da88cbad6d9b55183c', client_secret='48bb5ebd778f4223a2b0cdd3e9a3a66d')
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

               
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()

   
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
    searched_song_array = (pd.Series(searched_song_features).values / np.linalg.norm(pd.Series(searched_song_features).values)).reshape(1,songs_features.shape[1])

    # make the prediction
    prediction = knn.predict(searched_song_array)

    # Verify the searched song is not in the predictions.
    ten_most_similar_songs = songs.loc[prediction.argsort()[0][-11:]]
    if (ten_most_similar_songs['id'] == searched_song_id).any():
        ten_most_similar_songs = ten_most_similar_songs.drop(labels=ten_most_similar_songs['id'][ten_most_similar_songs['id'] == searched_song_id].index[0], axis=0)
    else:
        ten_most_similar_songs = ten_most_similar_songs[-10:]
    ten_most_similar_songs[['id','track_name','artist_name']]
    
    extendend_result = ten_most_similar_songs.append(searched_song_features,ignore_index=True)
    extendend_result['outcome'] = 1
    extendend_result.iloc[10,-1] = 0
    extendend_result = extendend_result[['acousticness','danceability','energy','instrumentalness',
                        'key', 'liveness', 'loudness','mode','speechiness', 'tempo',
                        'time_signature','valence','outcome']]

    extendend_result_for_plot = extendend_result
    # extendend_result_for_plot = extendend_result.T.reset_index()
    # extendend_result_for_plot = extendend_result_for_plot.rename({0: 'Recomendation 1',1: 'Recomendation 2',2: 'Recomendation 3',
    #                                                             3: 'Recomendation 4',4: 'Recomendation 5',5: 'Recomendation 6',
    #                                                             6: 'Recomendation 7',7: 'Recomendation 8',8: 'Recomendation 9',
    #                                                             9: 'Recomendation 10',10: 'Base Song'}, axis=1)

    print(extendend_result_for_plot)

    figs = go.Figure()

    # Add traces
    figs = make_subplots(rows=4, cols=3, shared_yaxes=True)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['danceability'], name='Danceability'),
                row=1, col=1)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['energy'], name='Energy'),
                row=1, col=2)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['liveness'], name='Liveness'),
                row=1, col=3)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['speechiness'], name='Speechiness'),
                row=2, col=1)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['acousticness'], name='Acousticness'),
                row=2, col=2)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['instrumentalness'], name='Instrumentalness'),
                row=2, col=3)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['loudness'], name='Loudness'),
                row=3, col=1)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['valence'], name='Valence'),
                row=3, col=2)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['tempo'], name='Tempo'),
                row=3, col=3)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['key'], name='Key'),
                row=4, col=1)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['mode'], name='Mode'),
                row=4, col=2)

    figs.add_trace(go.Scatter(x=extendend_result_for_plot.index, y=extendend_result_for_plot['time_signature'], name='Time signature'),
                row=4, col=3)


    figs.update_layout(height=800, width=800,
                    title_text="Difference of features between recomendations.")

    figs.update_xaxes(title_text="Songs")

    py.plot(figs, filename='subplots', sharing='public')



    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    return '<img src="data:image/png;base64,{}">'.format(plot_url)




if __name__ == '__main__':
    app.debug = True
    app.run()