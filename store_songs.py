import psycopg2

from dotenv import load_dotenv, find_dotenv
import os
from pull_data import SpotBot

load_dotenv()


# Database and Table creation

def create_tables():

    ''' Creates the table if it doesn't exists already.'''

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

        query_00 = '''
                CREATE TABLE IF NOT EXISTS songs (
                    id SERIAL,
                    song_id TEXT UNIQUE NOT NULL PRIMARY KEY,
                    song_name TEXT,
                    artists TEXT,
                    album TEXT,
                    danceability TEXT,
                    energy TEXT,
                    song_key TEXT,
                    loudness TEXT,
                    song_mode TEXT,
                    speechiness TEXT,
                    acousticness TEXT,
                    instrumentalness TEXT,
                    liveness TEXT,
                    valence TEXT,
                    tempo TEXT,
                    song_type TEXT,
                    duration_ms TEXT,
                    time_signature TEXT,
                    preview_url TEXT 
                );
        '''

        cursor.execute(query_00)

        connection.commit()
      
        query_0 = '''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL,
                    comb_id TEXT UNIQUE NOT NULL PRIMARY KEY,
                    username TEXT,
                    song_id TEXT
                );
        '''

        cursor.execute(query_0)

        connection.commit()
      

        return 'Success'
    
    except (Exception, psycopg2.Error) as error:
        print('Error verifying or creating the table.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')

    

def login_and_store_songs(username='spotify'):
    
    ''' Insert the songs to the database '''

    # Initialization of the class.

    sb = SpotBot()

    playlists_id = sb.pull_playlists(username=username)

    global_count = 1
    count = 1

    for playlist in playlists_id:
                     
        dictionaries = sb.pull_data(username=username,playlist=playlist)

        try:
            
            # Stablishes the connection to the db.

            connection = psycopg2.connect(user=os.environ.get('db_user'),
                                        password=os.environ.get('db_password'),
                                        host=os.environ.get('db_host'),
                                        port=os.environ.get('db_port'),
                                        database=os.environ.get('db_name'))

            # Create the cursor.

            cursor = connection.cursor()

            # Asign variables to be stored in the db.

            for dictionary in dictionaries:

                song_id = dictionary['song_id']
                song_name = dictionary['song_name']
                artists = dictionary['artists']
                album = dictionary['album']
                danceability = dictionary['danceability']
                energy = dictionary['energy']
                song_key = dictionary['key']
                loudness = dictionary['loudness']
                song_mode = dictionary['mode']
                speechiness = dictionary['speechiness']
                acousticness = dictionary['acousticness']
                instrumentalness = dictionary['instrumentalness']
                liveness = dictionary['liveness']
                valence = dictionary['valence']
                tempo = dictionary['tempo']
                song_type = dictionary['type']
                duration_ms = dictionary['duration_ms']
                time_signature = dictionary['time_signature']
                preview_url = dictionary['preview_url']

                comb_id = username + ' : ' + dictionary['song_id']
                username = username
                

                cursor.execute('''
                            SELECT song_id
                            FROM songs
                            WHERE song_id = %s
                ''', (song_id,))

                song_exists = cursor.fetchall()

                if not song_exists:

                    query_1 = '''
                            INSERT INTO songs (
                                song_id,
                                song_name,
                                artists,
                                album,
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
                                song_type,
                                duration_ms,
                                time_signature,
                                preview_url
                            )
                            VALUES (
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
                    cursor.execute(query_1,(song_id,song_name,artists,album,danceability,
                    energy,song_key,loudness,song_mode,speechiness,acousticness,
                    instrumentalness,liveness,valence,tempo,song_type,duration_ms,
                    time_signature,preview_url))

                    print('Song', song_id,' succesfully stored.')
                    print('Global count:', global_count,'; this count:', count)
                    count += 1
                    global_count += 1

                    connection.commit()

                else:
                                        
                    print(song_id, 'already in db, Global count:', global_count)
                    global_count += 1

                if username == 'spotify': 

                    print('Spotify is not an interesting username. LOL.')

                else:

                    cursor.execute('''
                            SELECT comb_id
                            FROM users
                            WHERE comb_id = %s
                    ''', (comb_id,))

                    comb_exists = cursor.fetchall()

                    if not comb_exists:

                        query_2 = '''
                                INSERT INTO users (
                                    comb_id,
                                    username,
                                    song_id
                                    )
                                    VALUES (
                                        %s,
                                        %s,
                                        %s
                                    );
                                '''
                        cursor.execute(query_2,(comb_id,username,song_id))

                        print('User preference', comb_id,' succesfully stored.')

                        connection.commit()

                    else:

                        print('User preference already in db.')


        except (Exception, psycopg2.Error) as error:
                    print('Error verifying or creating the table.')

        finally:

            if (connection):
                cursor.close()
                connection.close()
                print('Connection closed.')

        

        # return 'Success'

if __name__ == "__main__":
       
    create_tables()
    login_and_store_songs(username='Just_mike09')