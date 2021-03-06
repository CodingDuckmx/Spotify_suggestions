# Libraries Sections

import ast
import datetime
import joblib
import numpy as np
import os, os.path
import pandas as pd
import pickle
import psycopg2
import psycopg2.extras as psycopg2extras
import spotipy

from dotenv import load_dotenv, find_dotenv
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()

class SpotBot():

    def pull_playlists(self,username='spotify'):

        '''
        Pull from the db the list of playlists and songs aready in db.
        Pull the playlists list of the user, to determine if some id
        has to be dropped.
        Will pull the playlists of the user.
        Build a dictionary of playlists.
        In each playlist will be a list of dictionaries
        With the features of the songs of that playlist. 
        '''

        # Some metrics:

        # Playlist counter:

        playlist_counter = 0

        # Song counter

        song_counter = 0

        # Pull key information from the database.

        # Connection

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor.

        cursor = connection.cursor()

        # Pull data already in the db.

        cursor.execute('''
                        SELECT song_id
                        FROM songs;
        ''')

        db_songs_id_list = [item[0] for item in cursor.fetchall()]

        cursor.execute('''
                        SELECT playlist_id, added_date, last_modified_date
                        FROM playlists;
        ''')

        cursor_outcome = cursor.fetchall()

        if cursor_outcome:

            db_playlist_id_dct = {duet[0]:{'add_date':duet[1],'last_date':duet[2]} for duet in cursor_outcome}

        else:

            db_playlist_id_dct = {}

        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # A dictionary with playlists id as keys and list of their songs as items.
        playlists_dict = {}
        
        # Pull the data from the API
        api_outcomes = sp.user_playlists(username)

        # While (more) data available:
        while api_outcomes:

            # Trusting will get all playlist of the user
            for i in range(len(api_outcomes['items'])):
                
                playlist_id = api_outcomes['items'][i]['id']
                playlist_name = api_outcomes['items'][i]['name']

                # If the playlist is not already in the dict to be exported.
                if playlist_id not in playlists_dict:
                        
                    playlists_dict[playlist_id] = {}
                
                # Is the playlist in the db?
                if playlist_id not in db_playlist_id_dct:
                    
                    playlists_dict[playlist_id]['status'] = 'new'
                    print(f'***---> Adding the playlist: {playlist_name}. <---***')
                
                elif db_playlist_id_dct[playlist_id]['last_date'] < datetime.date.today():

                    playlists_dict[playlist_id]['status'] = 'updating'
                    print(f'***---> Updating the playlist: {playlist_name}. <---***')

                else:

                    playlists_dict[playlist_id]['status'] = 'passing on it'
                    print(f'***---> The playlist {playlist_name} is already updated in the db. <---***')

                # Add the playlist and its specification to a dictionary.
                # Prepare the list of the playlist

                playlists_dict[playlist_id]['name'] = playlist_name
                playlists_dict[playlist_id]['songs_list'] = []
                        
                # Pull the songs of that playlist

                api_outcomes_2 = sp.playlist_tracks(playlist_id=playlist_id)
  
                while api_outcomes_2:

                    # Trusting will get all songs of the playlist
                    for j in range(len(api_outcomes_2['items'])):

                        if api_outcomes_2['items'][j]['track']:

                            song_id = api_outcomes_2['items'][j]['track']['id']
                            song_name = api_outcomes_2['items'][j]['track']['name']

                            # Song dictionary for the info of the song

                            song_dict = {}

                            # song_dict = {key: None for key in ['song_id','song_name','artists','album','popularity','explicit','track_number',
                            # 'danceability', 'energy', 'song_key', 'loudness', 'song_mode', 'speechiness', 'acousticness', 'instrumentalness',
                            # 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'preview_url']}

                            song_dict['song_id'] = song_id

                            # Verify the song_id is not already in the db:

                            if song_id not in db_songs_id_list:                               

                                print(f'---> New song found, adding features of: {song_name} to the dictionary. <---')

                                try:

                                    song_dict['song_name'] = song_name
                                    song_dict['album'] = api_outcomes_2['items'][j]['track']['album']['id']
                                    song_dict['popularity'] = api_outcomes_2['items'][j]['track']['popularity']
                                    song_dict['explicit'] = api_outcomes_2['items'][j]['track']['explicit']
                                    song_dict['track_number'] = api_outcomes_2['items'][j]['track']['track_number']
                                    song_dict['duration_ms'] = api_outcomes_2['items'][j]['track']['duration_ms']
                                    song_dict['preview_url'] = api_outcomes_2['items'][j]['track']['preview_url']
                                    song_dict['year'] = api_outcomes_2['items'][j]['track']['album']['release_date'][:4]


                                    # Due a song could have more than one artist.
                                    # Build a list of artists 
                                    artists_list = []

                                    for k in range(len(api_outcomes_2['items'][j]['track']['artists'])):

                                        artists_list.append(api_outcomes_2['items'][j]['track']['artists'][k]['name'])

                                    song_dict['artists'] = artists_list

                                    # We have to use the API for the other missing features and artist.
                                    api_outcomes_3 = sp.audio_features(song_dict['song_id'])[0]
                                    song_dict['danceability'] = api_outcomes_3['danceability']
                                    song_dict['energy'] = api_outcomes_3['energy']
                                    song_dict['song_key'] = api_outcomes_3['key']
                                    song_dict['loudness'] = api_outcomes_3['loudness']
                                    song_dict['song_mode'] = api_outcomes_3['mode']
                                    song_dict['speechiness'] = api_outcomes_3['speechiness']
                                    song_dict['acousticness'] = api_outcomes_3['acousticness']
                                    song_dict['instrumentalness'] = api_outcomes_3['instrumentalness']
                                    song_dict['liveness'] = api_outcomes_3['liveness']
                                    song_dict['valence'] = api_outcomes_3['valence']
                                    song_dict['tempo'] = api_outcomes_3['tempo']
                                    song_dict['time_signature'] = api_outcomes_3['time_signature']

                                    playlists_dict[playlist_id]['songs_list'].append(song_dict)

                                    song_counter += 1

                                except:

                                    print(f"This song {song_name} has missing features in Spotify' API.")

                            # else:

                            #     print(f'This song {song_name} is already in the db.')

                        

                    if api_outcomes_2['next']:
                        api_outcomes_2 = sp.next(api_outcomes_2)
                    else:
                        api_outcomes_2 = None

                playlist_counter += 1


            if api_outcomes['next']:
                api_outcomes = sp.next(api_outcomes)
            else:
                api_outcomes = None
        
        # Close connection

        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

        print(f'{song_counter} songs embeded in {playlist_counter} playlist(s) from the user {username} were added to the db.')

        return playlists_dict

  

    def store_songs(self,username='spotify', similarity_threshold = 0.84):

        '''
        Takes a dictionary of playlists and its songs (including their features)
        and store them into the db.
        '''

        count_errors = 0

        if username == 'spotify':

            return 'Spotify is not a real user, try a real user/person.'

        pl_dict = self.pull_playlists(username=username)

        # Unfold the scalers and models :

        scalers_list = []
        models_list = []

        scalers_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/scalers')]
        models_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/models')]

        scalers_list_names.sort()
        models_list_names.sort()

        for name in scalers_list_names:

            scalers_list.append(joblib.load('song_matcher/scripts_and_pickes/pickles/scalers/' + name))

        for name in models_list_names:

            models_list.append(joblib.load('song_matcher/scripts_and_pickes/pickles/models/' + name))


### Starting: Considering to be removed ###
        # stdscaler = joblib.load('stdscalerpckl.pkl')
        # stdscaler_0 = joblib.load('stdscalerpckl_0.pkl')
        # stdscaler_1 = joblib.load('stdscalerpckl_1.pkl')
        # stdscaler_2 = joblib.load('stdscalerpckl_2.pkl')
        # stdscaler_3 = joblib.load('stdscalerpckl_3.pkl')

        # scalers_list = [stdscaler_0,stdscaler_1,stdscaler_2,stdscaler_3]

        # model = joblib.load('modelpckl.pkl')
        # model_0 = joblib.load('modelpckl_0.pkl')
        # model_1 = joblib.load('modelpckl_1.pkl')
        # model_2 = joblib.load('modelpckl_2.pkl')
        # model_3 = joblib.load('modelpckl_3.pkl')

        # models_list = [model_0,model_1,model_2,model_3]
### Finishing: considering to be removed ###
        
        
        # Stablish connection to the db

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor

        cursor = connection.cursor()

        for playlist_id in [x for x in pl_dict.keys()]:

            # Adding a brand new playlist to the db.
            if pl_dict[playlist_id]['status'] == 'new':

                query_values = (playlist_id,pl_dict[playlist_id]['name'],datetime.date.today(),
                          datetime.date.today(),[song['song_id'] for song in pl_dict[playlist_id]['songs_list']],[username])

                query_insert_new_pl = '''
                    INSERT INTO playlists (
                        playlist_id,
                        playlist_name,
                        added_date,
                        last_modified_date,
                        listed_songs,
                        followed_by
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    );
                '''

                try:

                    cursor.execute(query_insert_new_pl,query_values)

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(f'An error has occured when adding the playlist {pl_dict[playlist_id]["name"]} ({playlist_id}) to the db.')
                    print(f'Error: {e}')

            # We are updating a playlist.
            elif pl_dict[playlist_id]['status'] == 'updating':

                # First, pull the followed_by list, to add the possible new user

                cursor.execute('''
                    SELECT followed_by
                    FROM playlists
                    WHERE playlist_id = %s
                ''',(playlist_id,))

                cursor_outcome = cursor.fetchone()

                playlist_followers = str(cursor_outcome[0]).split(', ')

                playlist_followers[0] = playlist_followers[0][1:]
                playlist_followers[-1] = playlist_followers[-1][:-1]

                if username not in playlist_followers:
                
                    playlist_followers.append(username)

                listed_songs = [pl_dict[playlist_id]['songs_list'][i]['song_id'] for i in range(len(pl_dict[playlist_id]['songs_list']))]
        
                query_values = (datetime.date.today(),listed_songs,playlist_followers,playlist_id)

                query_update_pl = '''
                    UPDATE playlists
                    SET last_modified_date = %s,
                    listed_songs = %s,
                    followed_by = %s
                    WHERE playlist_id = %s
                '''

                try:

                    cursor.execute(query_update_pl,query_values)

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(f'An error has occured when updating the playlist {pl_dict[playlist_id]["name"]} ({playlist_id}) into the db.')
                    print(f'Error: {e}')

            # Does the user exist in the db?

            query_search_user = '''
                SELECT id
                FROM users
                WHERE username = %s
            '''

            cursor.execute(query_search_user,(username,))

            cursor_outcome = cursor.fetchone()

            following_playlists = [pl_id for pl_id in pl_dict.keys()]

            liked_songs = []

            for pl_id in following_playlists:

                for song_dict in pl_dict[pl_id]['songs_list']:

                    liked_songs.append(song_dict['song_id'])                 

            if not cursor_outcome:

                # Add the current user and their following playlists.

                query_add_user_n_following_pl = '''
                                        INSERT INTO users(
                                            username,
                                            liked_songs,
                                            following_playlists
                    )
                                        VALUES (
                                            %s,
                                            %s,
                                            %s
                    );
                '''

                try:

                    cursor.execute(query_add_user_n_following_pl,(username,liked_songs,following_playlists))

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(f'An error has occured when adding the user {username} to the db.')
                    print(f'Error: {e}')
            
            else:

                # Update the current following playlists of the user

                query_update_following_pl = '''
                    UPDATE users
                    SET liked_songs = %s,
                    following_playlists = %s
                    WHERE username = %s
                '''

                try:

                    cursor.execute(query_update_following_pl,(liked_songs, following_playlists, username))
                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(f'An error has occured when updating the user {username} into the db.')
                    print(f'Error: {e}')

            # Now, will insert the songs features to the db
            # For adding the liked_by data, I think I'll pospone this
            # willing to have another more efficient way.


            for song_dict in pl_dict[playlist_id]['songs_list']:

                # this means the song is not in the db yet. 
                if 'artists' in song_dict:

                    # Verify if the artists combination is in the database.

                    artists_str_list = '{' + str(song_dict['artists'])[1:-1].replace("'","") + '}'

                    cursor.execute('''
                                SELECT coded_artists
                                FROM songs
                                WHERE artists = %s;
                    ''',(artists_str_list,))

                    cursor_outcome = cursor.fetchone()

                    if cursor_outcome:

                        song_dict['coded_artists'] = cursor_outcome[0]

                    else:

                        # If the artist combination is new to the db
                        # we have to assing a code to this combination

                        cursor.execute('''
                                SELECT MAX(coded_artists)
                                FROM songs;
                        ''')

                        song_dict['coded_artists'] = int(cursor.fetchone()[0]) + 1

                    song_vector = np.array((song_dict['coded_artists'],song_dict['popularity'], song_dict['explicit'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
                        song_dict['speechiness'], song_dict['acousticness'], song_dict['instrumentalness'], song_dict['liveness'], song_dict['valence'], song_dict['tempo'], song_dict['duration_ms'], song_dict['year']),dtype=float)


                    song_dict['cluster'] = models_list[0].predict(scalers_list[0].transform(song_vector.reshape(1,-1)))[0].item()

                    song_dict['subcluster'] = models_list[song_dict['cluster']].predict(scalers_list[song_dict['cluster']].transform(song_vector.reshape(1,-1)))[0].item()

                    cursor.execute('''
                                SELECT coded_artists, popularity, explicit, danceability, energy, song_key, loudness, song_mode,
                                        speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, year
                                FROM songs
                                WHERE cluster = %s AND 
                                    subcluster = %s;
                                    ''',(song_dict['cluster'],song_dict['subcluster']))

                    cursor_outcome = cursor.fetchall()

                    db = np.array(cursor_outcome)

                    cursor.execute('''
                                SELECT song_id
                                FROM songs
                                WHERE cluster = %s AND 
                                    subcluster = %s;
                                    ''',(song_dict['cluster'],song_dict['subcluster']))

                    cursor_outcome_2 = cursor.fetchall()

                    if cursor_outcome_2:

                        songs_inside_subcluster = [item[0] for item in cursor_outcome_2]        

                        # Scale and normalize the vector song and the matrix of songs:

                        scaled_db = scalers_list[song_dict['cluster']].transform(db)
                        
                        norm_db = scaled_db / np.linalg.norm(scaled_db, axis=1, keepdims=True)
                        
                        scaled_song_vector = scalers_list[song_dict['cluster']].transform(song_vector.reshape(1,-1))

                        norm_song_vector = scaled_song_vector / np.linalg.norm(scaled_song_vector, keepdims=True)

                        similarities = norm_db.dot(norm_song_vector.T)

                        similarities = np.around(similarities,decimals=4)

                        similar_songs = {}
                        
                        # In this moment, the db will not be updated with similarities backward. 

                        for i, song_inside_id in enumerate(songs_inside_subcluster):

                            if song_inside_id != song_dict['song_id']:

                                similarity = similarities[i][0]

                                if similarity >= similarity_threshold:

                                    if similarity not in similar_songs:

                                        similar_songs[similarity] = []
                                    
                                    similar_songs[similarity].append(song_inside_id)

                        values_to_insert = (song_dict['song_id'],song_dict['song_name'], song_dict['artists'],song_dict['popularity'], song_dict['explicit'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
                                    song_dict['speechiness'], song_dict['acousticness'], song_dict['instrumentalness'], song_dict['liveness'], song_dict['valence'], song_dict['tempo'], song_dict['duration_ms'], song_dict['year'],song_dict['coded_artists'],str(similar_songs),
                                    song_dict['cluster'],song_dict['subcluster'])

                    else:
                        
                        count_errors += 1

                        print('#######################################################################')
                        print('#######################################################################')
                        print('#######################################################################')
                        print('Some songs has no similar songs dictionary. A refactoring is mandatory.')
                        print('#######################################################################')
                        print('#######################################################################')
                        print('#######################################################################')

                        values_to_insert = (song_dict['song_id'],song_dict['song_name'], song_dict['artists'],song_dict['popularity'], song_dict['explicit'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
                                    song_dict['speechiness'], song_dict['acousticness'], song_dict['instrumentalness'], song_dict['liveness'], song_dict['valence'], song_dict['tempo'], song_dict['duration_ms'], song_dict['year'],song_dict['coded_artists'],None,
                                    song_dict['cluster'],song_dict['subcluster'])

                    query_insert_song = '''
                        INSERT INTO songs (
                            song_id,
                            song_name,
                            artists,
                            popularity,
                            explicit,
                            danceability,
                            energy,
                            song_key,
                            loudness,
                            song_mode,
                            speechiness,
                            acousticness,
                            instrumentalness,
                            liveness,
                            valence,
                            tempo,
                            duration_ms,
                            year,
                            coded_artists,
                            similar_songs,
                            cluster,
                            subcluster
                            )
                        VALUES(
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s
                        ); 
                        '''

                    cursor.execute('''
                                SELECT id
                                FROM songs
                                WHERE song_id = %s;
                        ''',(song_dict['song_id'],))

                    cursor_outcome = cursor.fetchone()

                    if not cursor_outcome:

                        try:

                            cursor.execute(query_insert_song,values_to_insert)
                            
                            connection.commit()

                        except (Exception, psycopg2.Error) as error:

                            print(error)
                            print(f"There was an error with the song: {song_dict['song_id']}")
                            count_errors += 1

                        # finally:

                        #     print(f"There was an error with the song: {song_dict['song_id']}")
                        #     count_errors += 1

        # Finishes the connection to the db.

        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

        print(count_errors, 'errors found during the process.')

    def __db_clustering(self,array,max_n_clusters, songs_dict, looking_for, songs_id_list):

        '''
        Clusters the provided array.
        '''

        # Standarize the array.
        std_scaler = StandardScaler()

        std_array = std_scaler.fit_transform(array)

        ks = range(1,max_n_clusters)
        inertias = list()

        for k in ks:

            # Create a KMeans instance with k clusters.
            test_model = KMeans(n_clusters=k, random_state=42)

            # Fit the test model to the array
            test_model.fit(std_array)

            # Append the inertia to the list of inertias
            inertias.append(test_model.inertia_)

        kl = KneeLocator(ks,inertias,curve='convex',direction='decreasing')
        
        optimal_k = kl.elbow

        # Create a Kmeans instance with the optimal number of clusters.
        model = KMeans(n_clusters=optimal_k, random_state=42)

        # Create a Kmeans instance with k clusters
        model.fit_predict(std_array)

        # labels
        labels = model.labels_

        # Make a dictionary, dividing the song_features, according a which 
        # cluster it belongs.
        clustering_dict = dict()

        for i, label in enumerate(labels):

            if label not in clustering_dict:

                clustering_dict[label] = {'features': [], 'songs_ids': []}

            clustering_dict[label]['features'].append(array[i])
            clustering_dict[label]['songs_ids'].append(songs_id_list[i])

            songs_dict[songs_id_list[i]][looking_for] = label.item()

        # turn the list into an array, to be suitable for the next steps
        clustering_dict[label]['features'] = np.array(clustering_dict[label]['features'])

        return clustering_dict, songs_dict, model, std_scaler

    def __find_similar_songs(self, clustering_dict,cluster, subcluster, similarity_threshold = .89):

        features = clustering_dict['features'] 
        songs_ids = clustering_dict['songs_ids']

        songs_dict = {song_id:{'coded_artists':int(clustering_dict['features'][i][0]),'clustering':cluster,'subclustering':subcluster,'similar_songs':dict()} for i, song_id in enumerate(songs_ids)}

        std_scaler = StandardScaler()

        std_features = std_scaler.fit_transform(features)

        norm_features = std_features / np.linalg.norm(std_features, axis=1, keepdims=True)

        similarities_array = np.around(norm_features.dot(norm_features.T),4)

        for i, song_id in enumerate(songs_ids):

            # Limit the similarities to the top 20.
            top_similarities = []

            for j, similarity in enumerate(similarities_array[i]):

                if similarity >= similarity_threshold:

                    if i != j:

                        if similarity not in top_similarities:

                            top_similarities.append(similarity)

                            top_similarities.sort(reverse=True)

                            top_similarities = top_similarities[:20]

                        if similarity in top_similarities:

                            if similarity not in songs_dict[song_id]['similar_songs']:

                                songs_dict[song_id]['similar_songs'][similarity] = []

                            songs_dict[song_id]['similar_songs'][similarity].append(songs_ids[j])
                            
                        else:

                            if similarity in songs_dict[song_id]['similar_songs']:

                                del songs_dict[song_id]['similar_songs'][similarity]

        return songs_dict

    def update_similarities(self,similarity_threshold = 0.84):

        '''
        Pull all the db data.
        Divide the data into songs_id list and features array.
        
        '''

        # Erase the previous pickles and joblibs.

        scalers_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/scalers')]
        models_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/models')]
        dictionaries_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/similarities_dictionaries')]

        for file in scalers_list_names:

            os.remove('song_matcher/scripts_and_pickes/pickles/scalers/' + file)

        for file in models_list_names:

            os.remove('song_matcher/scripts_and_pickes/pickles/models/' + file)

        for file in dictionaries_list_names:

            os.remove('song_matcher/scripts_and_pickes/pickles/similarities_dictionaries/' + file)


        # Connection

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor.

        cursor = connection.cursor()        

        cursor.execute('''
                    SELECT song_id,
                    song_name,
                    artists,
                    popularity,
                    explicit,
                    danceability,
                    energy,
                    song_key,
                    loudness,
                    song_mode,
                    speechiness,
                    acousticness,
                    instrumentalness,
                    liveness,
                    valence,
                    tempo,
                    duration_ms,
                    year
                    FROM songs;
        ''')

        cursor_outcome = cursor.fetchall()

        songs_dict = {outcome[0]:{'song_name':outcome[1],'coded_artists':None,'clustering':None,'subclustering':None} for outcome in cursor_outcome}

        songs_id_list = [x for x in songs_dict.keys()]

        artists = [outcome[2] for outcome in cursor_outcome]

        le = LabelEncoder()

        coded_artists = [code.item() for code in le.fit_transform(artists)]

        songs_features = np.array([[0] + list(outcome[3:]) for outcome in cursor_outcome], dtype=float)

        songs_features[:,0] = coded_artists


        # Find the first clustering
        clustering_dict, songs_dict, model_instance, std_scaler_instance = self.__db_clustering(songs_features,len(songs_features[0])-2,songs_dict,'clustering',songs_id_list)
        
        # Pickle the model
        joblib.dump(model_instance, 'song_matcher/scripts_and_pickes/pickles/models/modelpckl.pkl')

        # Pickle the scaler
        joblib.dump(std_scaler_instance, 'song_matcher/scripts_and_pickes/pickles/scalers/stdscalerpckl.pkl')

        del model_instance
        del std_scaler_instance

        for i in clustering_dict.keys():

            songs_id_list = clustering_dict[i]['songs_ids']
            clustering_dict[i]['subclustering'], songs_dict, model_instance, std_scaler_instance = self.__db_clustering(clustering_dict[i]['features'],len(songs_features[0])-2,songs_dict,'subclustering',songs_id_list)

            # Pickle the model
            joblib.dump(model_instance, 'song_matcher/scripts_and_pickes/pickles/models/modelpckl_'+ str(i) + '.pkl')

            # Pickle the scaler
            joblib.dump(std_scaler_instance, 'song_matcher/scripts_and_pickes/pickles/scalers/stdscalerpckl_'+ str(i) + '.pkl')

            del model_instance
            del std_scaler_instance

            for j in clustering_dict[i]['subclustering'].keys():

                similarities_dict = self.__find_similar_songs(clustering_dict[i]['subclustering'][j], cluster = i, subcluster = j,similarity_threshold = similarity_threshold)

                # Pickle the final songs similarities dict.
                with open('song_matcher/scripts_and_pickes/pickles/similarities_dictionaries/songs_similarities_dict_' + str(i) + '_' + str(j) + '.p', 'wb') as f:
                    pickle.dump(similarities_dict,f)

                del similarities_dict

        # pickle.dump(songs_dict, open('songs_similarities_dict.p', 'wb'))

        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

    def update_db_similarities_from_pickles(self):

        # Stablish connection to the db

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor

        cursor = connection.cursor()

        dictionaries_list_names = [name for name in os.listdir('song_matcher/scripts_and_pickes/pickles/similarities_dictionaries')]

        for file in dictionaries_list_names:

            songs_dict = pickle.load( open( 'song_matcher/scripts_and_pickes/pickles/similarities_dictionaries/' + file , 'rb') )

            tpl_list = list()

            for song_id in songs_dict.keys():

                tpl_list.append((song_id,songs_dict[song_id]['coded_artists'],songs_dict[song_id]['clustering'].item(),songs_dict[song_id]['subclustering'].item(),str(songs_dict[song_id]['similar_songs'])))

            query = '''
                    UPDATE songs
                    SET coded_artists = data.coded_artists,
                        cluster = data.cluster,
                        subcluster = data.subcluster,
                        similar_songs = data.similar_songs
                    FROM (VALUES %s) AS data (song_id, coded_artists, cluster, subcluster, similar_songs)
                    WHERE data.song_id = songs.song_id;
            '''


            psycopg2extras.execute_values(cursor,query,tpl_list)

            connection.commit()



        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

    def pull_similar_songs_from_db(self,song_name,artist_name):

        # Stablish connection to the db 

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor

        cursor = connection.cursor()

        cursor.execute('''
                    SELECT similar_songs
                    FROM songs
                    WHERE UPPER(artists) LIKE UPPER(%s)
                    AND UPPER(song_name) LIKE UPPER(%s); 
                    ''', ('{%' + artist_name + '%}', '%' + song_name + '%' ))
     
        cursor_outcome = cursor.fetchall()

        if cursor_outcome:

            songs_similarities_list = []
            songs_similarities_dict = ast.literal_eval(cursor_outcome[0][0])
            
            for _, item in songs_similarities_dict.items():

                songs_similarities_list += item

            return songs_similarities_list
        
        return None

    # def find_similar_songs_from_pickles(self,song_uri,no_of_songs):

    #     '''
    #     Unwrap the pickle, order the similar songs.
    #     Return the most (no_of_songs) similar songs.
    #     '''

    #     songs_dict = pickle.load( open( 'songs_similarities_dict.p', 'rb') )

    #     recommended_songs = {}

    #     for similarity in sorted(songs_dict[song_uri]['similar_songs'].keys(), reverse=True)[:no_of_songs]:

    #         if similarity:

    #             for song_id in songs_dict[song_uri]['similar_songs'][similarity]:

    #                 if song_id in recommended_songs:

    #                     recommended_songs[song_id] = {'name':songs_dict[song_id]['song_name']}

    #     return recommended_songs

if __name__ == "__main__":
    
    #####
    '''
    some usernames:
    One playlist username = 8ppupllf9815lb0jzdj4dot8x
    imsammwalker
    Señora dejada lavando trastes: 12125120773
    Angel: 1279690720
    Ross: 22tbf4n6jbom3jvylao3jydbi
    Ceci: 12135358331
    '''
    #####

    username = 'rulo8817'

    username_list = ['8ppupllf9815lb0jzdj4dot8x','imsammwalker','12125120773','12124627674','12101981872','12143950223', 
                    '12135216693','12132279660','12100101104', 'denisselike','12178296539','22tbf4n6jbom3jvylao3jydbi',
                    '12135358331','12132061330','12122294869','12101615247','12137641658','zeves','1277827004',
                    '12127930580','12135965213','12120016735','12171104184','12139559971','12139559971','1288860546',
                    '1291615358','12158029326','adar.jalil','12100267030','12171773090','12136160439','1292180570',
                    '1298329795','pa7x3seb1p6var8o2p6tixyo7','ojleon','12122576634','12132823078','12143485078',
                    '12157507102','12168098060','1279839528','1280902170','1289035627','1295571934', 'alexgq87',
                    '2257gvsl2t64pweirfzkpsvdi', '22v4jwad7mewgwk3b7o5r4vua', 'wandererpublishing','1293411501',
                    'victorbesq', 'rulo8817','slinky_duck', '113805952','1144585077','1163174336','12100733765',
                    '12101718552','12123174013','12123368799','12131955846','12133697760','12137237711','12138092519',
                    '12141613881','12144828589','12150418460','12168038732','12179236172','12180114917','121854533',
                    '1276570602','1276617649','1276939404','1276989189','1277041201','1277086521','1277126844','1277434572',
                    '1277451946','1277768181','1278315967','1279736449','1280136979','1282342893','1283472810','1285722668',
                    '1287802341','1290109166','1291858007','1292798097','1296162383', '1298699622','1299017784','1299360987',
                    '223rmt26ivoovwnegprr5huza','225erdwr6gl3x26btp4vcwt7y','22gpnbdkp6ycgc2klu7qcfdua','22hkrtxz47q7vaetdqa3j74sy',
                    '22kclfntihs36oayactjyw4nq','22spuownsed6abxqadrw2k5wa','8841j15fsh70r7pe01urwdfui','adla1904','alecastroo',
                    'car10z','conchawasthere','dopetrackz','ecortesad','felizalr','gabadiel','gonzalez.estefan','hiperactiva','hiramverarivas',
                    'joelhzro','kjw30lf4wdewra8fuykah7gpx','ktydlr','mar-sosa','m%C3%A9xicov','p_moch','pepedelatorre','pop-rock-espa%C3%B1ol',
                    'rahea','yusarrikitiki','tandmproductionco','sustaitaana','spookymg','samuelbmora','rorocorleone','rahea',
                    '31kgbnb6ryhhkegmvsvs2vxxol3a','12146813420','12140114724','1298514046','2k4d5krsu7y3mxv0ax2yzlilr','6mwchrg8dfie78pqrrfhw5s8w']

    
    sb = SpotBot()

    # playlists_dict = sb.pull_playlists(username=username)
    # print("Test of the dict of playlists:", playlists_dict)
    # sb.store_songs()
    sb.store_songs(username=username)
    
    # for user in username_list:
        # sb.store_songs(username=user)
    # breakpoint()


    sb.update_similarities(similarity_threshold = 0.84)

    sb.update_db_similarities_from_pickles()

    # sb.find_similar_songs_from_pickle('4XTRWWCbB68WDeAHLfv2HP',10)