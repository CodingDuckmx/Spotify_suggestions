# Libraries Sections

import numpy as np
import os
import pandas as pd
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


    def pull_data(self,username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M'):

        client_credentials_manager = SpotifyClientCredentials(client_id=os.environ.get('client_id'), client_secret=os.environ.get('client_secret'))

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        dictionaries = []

        playlists = sp.user_playlists(username)

        column_names = ['song_id','song_name','artists','album','danceability', 'energy', 'key', 'loudness', 
        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'type',  'duration_ms', 'time_signature', 'preview_url']

        count = 1
        
        while playlists:

            for track, item in enumerate(playlists['items']):
                playlist = item['id']

                try:

                    dictionary = {k:np.nan for k in column_names}
                    dictionary['song_id'] = sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id']
                    dictionary['song_name'] = sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['name']
                    dictionary['artists'] = sp.user_playlist('spotify', playlist)['tracks']['items'][track]['track']['artists'][0]['name']
                    dictionary['album'] = sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['album']['name']
                    dictionary['danceability'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['danceability']
                    dictionary['energy'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['energy']
                    dictionary['key'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['key']
                    dictionary['loudness'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['loudness']
                    dictionary['mode'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['mode']
                    dictionary['speechiness'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['speechiness']
                    dictionary['acousticness'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['acousticness']
                    dictionary['instrumentalness'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['instrumentalness']
                    dictionary['liveness'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['liveness']
                    dictionary['valence'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['valence']
                    dictionary['tempo'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['tempo']
                    dictionary['type'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['type']
                    dictionary['duration_ms'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['duration_ms']
                    dictionary['time_signature'] = sp.audio_features(sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['id'])[0]['time_signature']
                    dictionary['preview_url'] = sp.user_playlist('spotify',playlist)['tracks']['items'][track]['track']['preview_url']
                    dictionaries.append(dictionary)

                    print(dictionary['song_id'],' song added to dictionary; count:', count,'.')
                    count +=1


                except:
                    pass

            if playlists['next']:
                playlists = sp.next(playlists)
            else:
                playlists = None

        print('List of dictionaries ready to be added to the db.')

        return dictionaries




if __name__ == "__main__":
    
    sb = SpotBot()
    playlists_id = sb.pull_playlists(username='spotify')
    print("Test of list of playlists id's:", playlists_id[:20])
    dictionary = sb.pull_data(username='spotify',playlist='37i9dQZF1DXcBWIGoYBM5M',track=0)
    print("Test of result dictionaries: ",dictionaries)