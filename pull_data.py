# Libraries Sections

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

        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        playlists_id = []

        playlists_items = sp.user_playlists(username)

        while playlists_items:
            for playlist in range(len(playlists_items['items'])):
                playlists_id.append(playlists_items['items'][playlist]['id'])

            if playlists_items['next']:
                playlists_items = sp.next(playlists_items)
            else:
                playlists_items = None
        
        return playlists_id

    def pull_songs_ids(self,username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M'):

        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        dictionaries = []

        # Pulls all the playlists of the users.
        playlists = sp.user_playlists(username)

        breakpoint()

        return 


    def pull_data(self,username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M'):

        # First pull all the songs already in the db.

        try:
        
            # Stablishes the connection to the db.

            connection = psycopg2.connect(user=os.environ.get('db_user'),
                                        password=os.environ.get('db_password'),
                                        host=os.environ.get('db_host'),
                                        port=os.environ.get('db_port'),
                                        database=os.environ.get('db_name'))

            print('DB Connection:', connection)

            # Create the cursor.

            cursor = connection.cursor()
            print('DB Cursor:', cursor)

            cursor.execute('''
                        SELECT song_id
                        from songs
            ''')

            songs_list = cursor.fetchall()
            songs_list = [tpl[0] for tpl in songs_list]

 

        except (Exception, psycopg2.Error) as error:
            print('Error pulling the songs list from the db.')

        finally:

            if (connection):
                cursor.close()
                connection.close()
                print('Connection closed.')



        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        dictionaries = []

        # Pulls all the playlists of the users.
        playlists = sp.user_playlists(username)

        column_names = ['song_id','song_name','artists','album','popularity','explicit','track_number',
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
         'liveness', 'valence', 'tempo', 'type',  'duration_ms', 'time_signature', 'preview_url']

        count = 1
        
        # Playlists is like a linked list. Each one is linked with the following.
        # Once we are done with the first, we'll jump to the next.        
        while playlists:

            # Playlists are dictionaries with keys:
            # ['href', 'items', 'limit', 'next', 'offset', 'previous', 'total']
            # We are looping through the first list of items.
            for playlist in playlists['items']:
                
                # We pull the first playlist id of the list of playlists.
                playlist_id = playlist['id']

                # Number of tracks in this playlist:
                num_of_tracks_in_the_playlist = sp.playlist_tracks(playlist_id=playlist_id)['total']

                # The tracks come in bunches of at most 100 tracks.
                # We initialize a counter at zero.

                start = 0

                while start < num_of_tracks_in_the_playlist:

                    num_of_tracks_this_round = len(sp.playlist_tracks(playlist_id=playlist_id,offset=start)['items'])

                    for i in range(num_of_tracks_this_round):

                        try:

                            base_path = sp.playlist_tracks(playlist_id=playlist_id,offset=start)['items'][i]['track']
                            dictionary = {k:np.nan for k in column_names}

                            song_id = base_path['id']

                            if song_id not in songs_list:

                                dictionary['song_id'] = song_id
                                dictionary['song_name'] = base_path['name']
                                dictionary['artists'] = base_path['artists'][0]['name']
                                dictionary['album'] = base_path['album']['name']
                                dictionary['popularity'] = base_path['popularity']
                                dictionary['explicit'] = base_path['explicit']
                                dictionary['track_number'] = base_path['track_number']
                                dictionary['danceability'] = sp.audio_features(song_id)[0]['danceability']
                                dictionary['energy'] = sp.audio_features(song_id)[0]['energy']
                                dictionary['key'] = sp.audio_features(song_id)[0]['key']
                                dictionary['loudness'] = sp.audio_features(song_id)[0]['loudness']
                                dictionary['mode'] = sp.audio_features(song_id)[0]['mode']
                                dictionary['speechiness'] = sp.audio_features(song_id)[0]['speechiness']
                                dictionary['acousticness'] = sp.audio_features(song_id)[0]['acousticness']
                                dictionary['instrumentalness'] = sp.audio_features(song_id)[0]['instrumentalness']
                                dictionary['liveness'] = sp.audio_features(song_id)[0]['liveness']
                                dictionary['valence'] = sp.audio_features(song_id)[0]['valence']
                                dictionary['tempo'] = sp.audio_features(song_id)[0]['tempo']
                                dictionary['type'] = sp.audio_features(song_id)[0]['type']
                                dictionary['duration_ms'] = sp.audio_features(song_id)[0]['duration_ms']
                                dictionary['time_signature'] = sp.audio_features(song_id)[0]['time_signature']
                                dictionary['preview_url'] = base_path['preview_url']
                                
                                
                                dictionaries.append(dictionary)

                                print(dictionary['song_id'],' song added to dictionary; count:', count,'.')
                                count +=1

                            else:

                                print(song_id,'already in db.')

                        except:
                            
                            print('Error building the dictionary. This song might not have audio features.')
                            print(song_id)
                            print(dictionary['song_name'])

                            # breakpoint()



                    start += 100

            if playlists['next']:
                playlists = sp.next(playlists)
            else:
                playlists = None

        print('List of dictionaries ready to be added to the db.')

        # breakpoint()

        return dictionaries




if __name__ == "__main__":
    
    sb = SpotBot()
    # playlists_id = sb.pull_playlists(username='slinky_duck')
    # print("Test of list of playlists id's:", playlists_id[:20])
    sb.pull_songs_ids(username='slinky_duck',playlist='395EBEpBl4C2eOOnMHsj3i')
    # dictionary = sb.pull_data(username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M')
    # print("Test of result dictionaries: ",dictionaries)

