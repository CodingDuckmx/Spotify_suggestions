# Libraries Sections

import ast
import datetime
import joblib
import numpy as np
import os
import pandas as pd
import psycopg2
import spotipy

from dotenv import load_dotenv, find_dotenv
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

        if cursor.fetchall():

            db_playlist_id_dct = {duet[0]:{'add_date':duet[1],'last_date':duet[2]} for duet in cursor.fetchall()}

        else:

            db_playlist_id_dct = {}

        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # A dictionary with playlists id as keys and list of their songs as items.
        playlists_dict = {'list_of_live_pl':[],'some_of_liked_songs':[]}

        # Pull the data from the API
        api_outcomes = sp.user_playlists(username)

        # While (more) data available:
        while api_outcomes:

            # Trusting will get all playlist of the user
            for i in range(len(api_outcomes['items'])):
                
                playlist_id = api_outcomes['items'][i]['id']
                playlist_name = api_outcomes['items'][i]['name']

                # Add the playlist_id to the index of live playlists:

                playlists_dict['list_of_live_pl'].append(playlist_id)

                # verify if the playlist is already in the database
                # and if so, the last modification was up to yesterday.

                if playlist_id not in db_playlist_id_dct or db_playlist_id_dct[playlist_id]['last_date'] < datetime.date.today():

                    # Add the playlist and its specification to a dictionary.

                    if playlist_id not in playlists_dict:

                        # Prepare the list of the playlist

                        playlists_dict[playlist_id] = {}
                        playlists_dict[playlist_id]['name'] = playlist_name
                        playlists_dict[playlist_id]['songs_list'] = []
                        playlists_dict[playlist_id]['new'] = True

                        # If we are updating the playlist, change the value of new.

                        if db_playlist_id_dct:
                            
                            if playlist_id in db_playlist_id_dct:

                                playlists_dict[playlist_id]['new'] = False

                                print(f'***---> Updating the playlist: {playlist_name}. <---***')

                        else:

                            print(f'***---> Adding the playlist: {playlist_name}. <---***')


                    # Pull the songs of that playlist

                    api_outcomes_2 = sp.playlist_tracks(playlist_id=playlist_id)

                    
                    while api_outcomes_2:

                        # Trusting will get all songs of the playlist
                        for j in range(len(api_outcomes_2['items'])):

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

                                song_counter += 1

                            else:

                                print(f'This song {song_name} is already in the db.')

                            playlists_dict['some_of_liked_songs'].append(song_id)
                            playlists_dict[playlist_id]['songs_list'].append(song_dict)

                        if api_outcomes_2['next']:
                            api_outcomes_2 = sp.next(api_outcomes_2)
                        else:
                            api_outcomes_2 = None

                    playlist_counter += 1

                else:

                    print(f'This playlist {playlist_name} is relatively new to the db.')

            if api_outcomes['next']:
                api_outcomes = sp.next(api_outcomes)
            else:
                api_outcomes = None
        

        # Close connection

        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

        print(f'{song_counter} songs embeded in {playlist_counter} playlist(s) were added to the db.')



        return playlists_dict

  

    def store_songs(self,username='spotify',pl_dict=None):

        '''
        Takes a dictionary of playlists and its songs (including their features)
        and store them into the db.
        '''
        if username == 'spotify':

            return 'Spotify is not a real user, try a real user/person.'

        if not pl_dict:

            pl_dict = self.pull_playlists(username=username)

        # Unfold the scalers and models :

        stdscaler = joblib.load('stdscalerpckl.pkl')
        stdscaler_0 = joblib.load('stdscalerpckl_0.pkl')
        stdscaler_1 = joblib.load('stdscalerpckl_1.pkl')
        stdscaler_2 = joblib.load('stdscalerpckl_2.pkl')
        stdscaler_3 = joblib.load('stdscalerpckl_3.pkl')

        scalers_list = [stdscaler_0,stdscaler_1,stdscaler_2,stdscaler_3]

        model = joblib.load('modelpckl.pkl')
        model_0 = joblib.load('modelpckl_0.pkl')
        model_1 = joblib.load('modelpckl_1.pkl')
        model_2 = joblib.load('modelpckl_2.pkl')
        model_3 = joblib.load('modelpckl_3.pkl')

        models_list = [model_0,model_1,model_2,model_3]

        # Stablish connection to the db

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        # Create cursor

        cursor = connection.cursor()

        for playlist_id in [x for x in pl_dict.keys()][2:]:

            # Adding a brand new playlist to the db.
            if pl_dict[playlist_id]['new']:

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

                    print(e)

            # We are updating a playlist.
            else:

                # First, pull the followed_by list, to add the possible new user

                cursor.execute('''
                    SELECT followed_by
                    FROM playlists
                    WHERE playlist_id = %s
                ''',(playlist_id,))

                playlist_followers = ast.literal_eval(cursor.fetchone()[0])
                
                if username not in playlist_followers:
                
                    playlist_followers.append(username)

        
                query_values = (datetime.date.today(),[song['song_id'] for song in pl_dict[playlist_id]['songs_list']],playlist_followers,playlist_id)

                query_update_not_new_pl = '''
                    UPDATE playlists
                    SET last_modified_date = %s,
                    listed_songs = %s,
                    followed_by = %s
                    WHERE playlist_id = %s
                '''

                try:

                    cursor.execute(query_update_not_new_pl,query_values)

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(e)

            # Does the user exist in the db?

            query_search_user = '''
                SELECT id
                FROM users
                WHERE username = %s
            '''

            cursor.execute(query_search_user,(username,))

            if not cursor.fetchone():

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

                    cursor.execute(query_add_user_n_following_pl,(username,pl_dict['some_of_liked_songs'],pl_dict['list_of_live_pl']))

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(e)
            
            else:

                # Update the current following playlists of the user

                query_update_following_pl = '''
                    UPDATE users
                    SET following_playlists = %s
                    WHERE username = %s
                '''

                try:

                    cursor.execute(query_update_following_pl,(pl_dict['list_of_live_pl'],username))

                    connection.commit()                  

                except psycopg2.OperationalError as e: 

                    print(e)

            # Now, will insert the songs features to the db
            # For adding the liked_by data, I think I'll pospone this
            # willing to have another more efficient way.


            for song_dict in pl_dict[playlist_id]['songs_list']:

                # this means the song is not in the db yet. 
                if 'artists' in song_dict:

                    # Verify if the artists combination is in the database.

                    cursor.execute('''
                                SELECT coded_artists
                                FROM songs
                                WHERE artists = %s;
                    ''',(str(song_dict['artists'])[1:-1],))
                    try:

                        if cursor.fetchone():
                            
                            

                            song_dict['coded_artists'] = cursor.fetchone()[0]

                        else:

                            # If the artist combination is new to the db
                            # we have to assing a code to this combination

                            cursor.execute('''
                                    SELECT MAX(coded_artists)
                                    FROM songs;
                            ''')

                            song_dict['coded_artists'] = int(cursor.fetchone()[0]) + 1



                        song_vector = np.array((song_dict['coded_artists'],song_dict['popularity'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
                            song_dict['speechiness'], song_dict['acousticness'], song_dict['instrumentalness'], song_dict['liveness'], song_dict['valence'], song_dict['tempo'], song_dict['duration_ms'], song_dict['year']))

                    except:

                        cursor.close()
                        connection.close()
                        breakpoint()

                    song_dict['cluster'] = model.predict(stdscaler.transform(song_vector.reshape(1,-1)))[0].item()

                    song_dict['subcluster'] = models_list[song_dict['cluster']].predict(scalers_list[song_dict['cluster']].transform(song_vector.reshape(1,-1)))[0].item()


                    cursor.execute('''
                                SELECT coded_artists, popularity, danceability, energy, song_key, loudness, song_mode,
                                        speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, year
                                FROM songs
                                WHERE cluster = %s AND 
                                    subcluster = %s;
                                    ''',(song_dict['cluster'],song_dict['subcluster']))

                    db = np.array(cursor.fetchall())

                    cursor.execute('''
                                SELECT song_id
                                FROM songs
                                WHERE cluster = %s AND 
                                    subcluster = %s;
                                    ''',(song_dict['cluster'],song_dict['subcluster']))

                    songs_inside_subcluster = [item[0] for item in cursor.fetchall()]        

                    # Scale and normalize the vector song and the matrix of songs:

                    scaled_db = scalers_list[song_dict['cluster']].transform(db)
                    
                    norm_db = scaled_db / np.linalg.norm(scaled_db)
                    
                    scaled_song_vector = scalers_list[song_dict['cluster']].transform(song_vector.reshape(1,-1))

                    norm_song_vector = scaled_song_vector / np.linalg.norm(scaled_song_vector)

                    similarities = norm_db.dot(norm_song_vector.T)

                    similar_songs = {}
                    
                    # In this moment, the db will not be updated with similarities backward. 

                    for i, song_inside_id in enumerate(songs_inside_subcluster):

                        if song_inside_id != song_dict['song_id']:

                            if similarities[i] >= 0.75:

                                if similarities[i] not in similar_songs:

                                    similar_songs[similarities[i]] = []
                                
                                similar_songs[similarities[i]].append(song_inside_id)

                    if similar_songs:

                        values_to_insert = (song_dict['song_id'],song_dict['song_name'], str(song_dict['artists'])[1:-1],song_dict['popularity'], song_dict['explicit'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
                                    song_dict['speechiness'], song_dict['acousticness'], song_dict['instrumentalness'], song_dict['liveness'], song_dict['valence'], song_dict['tempo'], song_dict['duration_ms'], song_dict['year'],song_dict['coded_artists'],similar_songs,
                                    song_dict['cluster'],song_dict['subcluster'])

                    else:

                        values_to_insert = (song_dict['song_id'],song_dict['song_name'], str(song_dict['artists'])[1:-1],song_dict['popularity'], song_dict['explicit'], song_dict['danceability'], song_dict['energy'], song_dict['song_key'], song_dict['loudness'], song_dict['song_mode'],
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

                    try:

                        cursor.execute(query_insert_song,values_to_insert)
                        
                        connection.commit()

                    except (Exception, psycopg2.Error) as error:

                        print(f"There was an error with the song: {song_dict['song_id']}")
                        print(error)





        # Finishes the connection to the db.

        if connection:

            cursor.close()
            connection.close()
            print('Connection closed.')

        # breakpoint()

        ###########################
        #### Being deprecated #####
        ####     starts       #####
        ###########################

    # def pull_songs_ids(self,username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M'):

    #     client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

    #     sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    #     dictionaries = []

    #     # Pulls all the playlists of the users.
    #     playlists = sp.user_playlists(username)

    #     breakpoint()

    #     return 


    # def pull_data(self,username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M'):

    #     # First pull all the songs already in the db.

    #     try:
        
    #         # Stablishes the connection to the db.

    #         connection = psycopg2.connect(user=os.environ.get('db_user'),
    #                                     password=os.environ.get('db_password'),
    #                                     host=os.environ.get('db_host'),
    #                                     port=os.environ.get('db_port'),
    #                                     database=os.environ.get('db_name'))

    #         print('DB Connection:', connection)

    #         # Create the cursor.

    #         cursor = connection.cursor()
    #         print('DB Cursor:', cursor)

    #         cursor.execute('''
    #                     SELECT song_id
    #                     from songs
    #         ''')

    #         songs_list = cursor.fetchall()
    #         songs_list = [tpl[0] for tpl in songs_list]

 

    #     except (Exception, psycopg2.Error) as error:
    #         print('Error pulling the songs list from the db.')

    #     finally:

    #         if (connection):
    #             cursor.close()
    #             connection.close()
    #             print('Connection closed.')



    #     client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

    #     sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    #     dictionaries = []

    #     # Pulls all the playlists of the users.
    #     playlists = sp.user_playlists(username)

    #     column_names = ['song_id','song_name','artists','album','popularity','explicit','track_number',
    #     'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    #      'liveness', 'valence', 'tempo', 'type',  'duration_ms', 'time_signature', 'preview_url']

    #     count = 1
        
    #     # Playlists is like a linked list. Each one is linked with the following.
    #     # Once we are done with the first, we'll jump to the next.        
    #     while playlists:

    #         # Playlists are dictionaries with keys:
    #         # ['href', 'items', 'limit', 'next', 'offset', 'previous', 'total']
    #         # We are looping through the first list of items.
    #         for playlist in playlists['items']:
                
    #             # We pull the first playlist id of the list of playlists.
    #             playlist_id = playlist['id']

    #             # Number of tracks in this playlist:
    #             num_of_tracks_in_the_playlist = sp.playlist_tracks(playlist_id=playlist_id)['total']

    #             # The tracks come in bunches of at most 100 tracks.
    #             # We initialize a counter at zero.

    #             start = 0

    #             while start < num_of_tracks_in_the_playlist:

    #                 num_of_tracks_this_round = len(sp.playlist_tracks(playlist_id=playlist_id,offset=start)['items'])

    #                 for i in range(num_of_tracks_this_round):

    #                     try:

    #                         base_path = sp.playlist_tracks(playlist_id=playlist_id,offset=start)['items'][i]['track']
    #                         dictionary = {k:np.nan for k in column_names}

    #                         song_id = base_path['id']

    #                         if song_id not in songs_list:

    #                             dictionary['song_id'] = song_id
    #                             dictionary['song_name'] = base_path['name']
    #                             dictionary['artists'] = base_path['artists'][0]['name']
    #                             dictionary['album'] = base_path['album']['name']
    #                             dictionary['popularity'] = base_path['popularity']
    #                             dictionary['explicit'] = base_path['explicit']
    #                             dictionary['track_number'] = base_path['track_number']
    #                             dictionary['danceability'] = sp.audio_features(song_id)[0]['danceability']
    #                             dictionary['energy'] = sp.audio_features(song_id)[0]['energy']
    #                             dictionary['key'] = sp.audio_features(song_id)[0]['key']
    #                             dictionary['loudness'] = sp.audio_features(song_id)[0]['loudness']
    #                             dictionary['mode'] = sp.audio_features(song_id)[0]['mode']
    #                             dictionary['speechiness'] = sp.audio_features(song_id)[0]['speechiness']
    #                             dictionary['acousticness'] = sp.audio_features(song_id)[0]['acousticness']
    #                             dictionary['instrumentalness'] = sp.audio_features(song_id)[0]['instrumentalness']
    #                             dictionary['liveness'] = sp.audio_features(song_id)[0]['liveness']
    #                             dictionary['valence'] = sp.audio_features(song_id)[0]['valence']
    #                             dictionary['tempo'] = sp.audio_features(song_id)[0]['tempo']
    #                             dictionary['type'] = sp.audio_features(song_id)[0]['type']
    #                             dictionary['duration_ms'] = sp.audio_features(song_id)[0]['duration_ms']
    #                             dictionary['time_signature'] = sp.audio_features(song_id)[0]['time_signature']
    #                             dictionary['preview_url'] = base_path['preview_url']
                                
                                
    #                             dictionaries.append(dictionary)

    #                             print(dictionary['song_id'],' song added to dictionary; count:', count,'.')
    #                             count +=1

    #                         else:

    #                             print(song_id,'already in db.')

    #                     except:
                            
    #                         print('Error building the dictionary. This song might not have audio features.')
    #                         print(song_id)
    #                         print(dictionary['song_name'])

    #                         # breakpoint()



    #                 start += 100

    #         if playlists['next']:
    #             playlists = sp.next(playlists)
    #         else:
    #             playlists = None

    #     print('List of dictionaries ready to be added to the db.')

    #     # breakpoint()

    #     return dictionaries


        ###########################
        #### Being deprecated #####
        #####     ends       ######
        ###########################

if __name__ == "__main__":
    
    #####
    '''
    some usernames:
    One playlist username = 8ppupllf9815lb0jzdj4dot8x
    imsammwalker

    '''
    #####

    sb = SpotBot()
    # playlists_dict = sb.pull_playlists(username='8ppupllf9815lb0jzdj4dot8x')
    # print("Test of the dict of playlists:", playlists_dict)
    sb.store_songs(username='slinky_duck')
    # breakpoint()
    # sb.pull_songs_ids(username='slinky_duck',playlist='395EBEpBl4C2eOOnMHsj3i')
    # dictionary = sb.pull_data(username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M')
    # print("Test of result dictionaries: ",dictionaries)

