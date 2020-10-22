from app import db

class User(db.Model):

    __tablename__ = 'users'

    song_id = db.Column(db.String(255), primary_key=True)
    song_name = db.Column(db.String(255), nullable=False)
    artists = db.Column(db.String(255), nullable=False)
    album = db.Column(db.String(255), nullable=False)
    danceability = db.Column(db.String(255), nullable=False)
    energy = db.Column(db.String(255), nullable=False)
    key = db.Column(db.String(255), nullable=False)
    loudness = db.Column(db.String(255), nullable=False)
    mode = db.Column(db.String(255), nullable=False)
    speechiness = db.Column(db.String(255), nullable=False)
    acousticness = db.Column(db.String(255), nullable=False)
    instrumentalness = db.Column(db.String(255), nullable=False)
    liveness = db.Column(db.String(255), nullable=False)
    valence = db.Column(db.String(255), nullable=False)
    tempo = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(255), nullable=False)
    duration_ms = db.Column(db.String(255), nullable=False)
    time_signature = db.Column(db.String(255), nullable=False)
    preview_url = db.Column(db.String(255), nullable=False)

# class SpotifyModel():

#     '''
#     Finds similar songs to a given song. 
#     '''

def SpotifySimilarities(unknown):


    # Pulls the info form the database.
    conn = psycopg2.connect("dbname='phmcuozt' user='phmcuozt' host='drona.db.elephantsql.com' password='Hl4xzpVOZxiQ9af4kH5bavoEHIx7z3hn'")
    songs = pd.read_sql_query('SELECT * FROM spotify_table', conn)
    conn.close()

    # Searching into Spotify for the first result of the query.
    result = sp.search(q=f'artist:{artist_name} track:{track_name}')
    api_features_results = sp.audio_features(searched_song_id)

    searched_song_info = {'artist_name' : result['tracks']['items'][0]['artists'][0]['name'], 
                        'track_name' : result['tracks']['items'][0]['name'],
                        'id' : result['tracks']['items'][0]['id'],
                        'popularity': result['tracks']['items'][0]['popularity'],
                        'danceability': api_features_results[0]['danceability'],
                        'energy':api_features_results[0]['energy'],
                        'key':api_features_results[0]['key'],
                        'loudness':api_features_results[0]['loudness'],
                        'mode':api_features_results[0]['mode'],
                        'speechiness':api_features_results[0]['speechiness'],
                        'acousticness':api_features_results[0]['acousticness'],
                        'instrumentalness':api_features_results[0]['instrumentalness'],
                        'liveness':api_features_results[0]['liveness'],
                        'valence':api_features_results[0]['valence'],
                        'tempo':api_features_results[0]['tempo'],
                        'time_signature':api_features_results[0]['time_signature']}
    
    # apend the result to be sure the values are in the same order than the rest
    songs = songs.append(searched_song_info, ignore_index=True)

    # define the features to be measured.
    features = ['acousticness','danceability','energy','instrumentalness',
                      'key', 'liveness', 'loudness','mode','speechiness', 'tempo',
                      'time_signature','valence','popularity']
    songs[features] = songs[features].astype(float)

    # Define the base vector.
    searched_song = songs[features].iloc[-1,:]

    # Founding the cosine of the angule between them.

    def cosdist(chosen_song,all_songs):
        ''' Dot product of each song with the chosen song, and normilize to get the cosine between the vectors '''

        normalize_song = chosen_song / np.linalg.norm(chosen_song, keepdims=True)
        normalize_all = all_songs / np.linalg.norm(all_songs, axis=1, keepdims=True)
        dotproduct = normalize_song.dot(normalize_all.T)

        return dotproduct

    # appending all distances between the searched song and the songs of the database.
    distances_list = []
    distances = cosdist(searched_song,songs_features)
    for i in range(len(songs_features)):
        distances_list.append((i,distances[i]))

    distances_list.sort(key=lambda x: x[1])

    # pull a bunch of similar songs
    similar_songs_index = [distances_list[-42:][i][0] for i in range(len(distances_list[-42:]))]

    # Drop versions of the same song with the same title.
    similar_songs = songs.loc[similar_songs_index]
    similar_songs[similar_songs.duplicated(['artist_name', 'track_name'], keep='first')]

    # Drop the searched song from the similar songs.
    if (similar_songs['id'] == chosen_song['id']).any():
        similar_songs = similar_songs.drop(labels=similar_songs[similar_songs['id'] == chosen_song['id']].index[0],axis=0)

    # Define the top 10 most similar
    ten_more_similar = similar_songs[-10:]

    
