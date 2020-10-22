import os
import psycopg2

from dotenv import load_dotenv, find_dotenv
from pull_data import SpotBot

load_dotenv()


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
                popularity = dictionary['popularity']

                if dictionary['explicit'] == 'True':

                    explicit = True
                
                else:

                    explicit = False

                track_number = dictionary['track_number']
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
                                popularity,
                                explicit,
                                track_number,
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
                                %s,
                                %s,
                                %s,
                                %s
                            );
                    '''
                    cursor.execute(query_1,(song_id,song_name,artists,album,popularity,
                    explicit,track_number,danceability,energy,song_key,loudness,song_mode,
                    speechiness,acousticness,instrumentalness,liveness,valence,tempo,
                    song_type,duration_ms,time_signature,preview_url))

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
                            SELECT username, liked_songs, liked_artists
                            FROM users
                            WHERE username = %s
                    ''', (username,))

                    user_exists = cursor.fetchone()

                    if not user_exists:

                        query_2 = '''
                                INSERT INTO users (
                                    username
                                    )
                                    VALUES (
                                        %s
                                    );
                                '''
                        cursor.execute(query_2,(username,))

                        print('User ', username,' succesfully added to the users table.')

                        connection.commit()

                    try:

                        if not user_exists[1]:

                            query_3 = '''
                                    UPDATE users
                                    SET liked_songs = %s
                                    WHERE username = %s;
                                    '''

                            cursor.execute(query_3,(song_id,username))

                            print('User ', username,' succesfully added to the users table.')

                            connection.commit()

                        else:

                            # Is the outcome a list?

                            if ',' in user_exists[1]:
                                
                                # Turn the string into a list

                                old_list = user_exists[1].split(',')

                                # If the new song id not in the list, add it.

                                if song_id not in old_list:

                                    new_list = user_exists[1] + ',' + str(song_id)

                                # Else, keep the old list that way.

                                else:

                                    new_list = user_exists[1]

                            # Else, verify if the stored value is the same song id.

                            else:

                                if song_id == user_exists[1]:

                                    new_list = user_exists[1]
                                
                                # If not, add it.

                                else:

                                    new_list = user_exists[1] + ',' + str(song_id)

                            # Insert it to the db

                            query_4 = '''
                                    UPDATE users
                                    SET liked_songs = %s
                                    WHERE username = %s;
                                    '''
                            cursor.execute(query_4,(new_list,username))

                            print('User ', username,' succesfully added to the users table.')

                            connection.commit()

                        if not user_exists[2]:

                            query_5 = '''
                                    UPDATE users
                                    SET liked_artists = %s
                                    WHERE username = %s;
                                    '''

                            cursor.execute(query_5,(artists,username))

                            print('Artist ', artists,' succesfully added to user,', username ,' row.')

                            connection.commit()

                        else:

                            # Is the outcome a list?

                            if ',' in user_exists[2]:
                                
                                # Turn the string into a list

                                old_list = user_exists[2].split(',')

                                # If the new song id not in the list, add it.

                                if artists not in old_list:

                                    new_list = user_exists[2] + ',' + str(artists)

                                # Else, keep the old list that way.

                                else:

                                    new_list = user_exists[2]

                            # Else, verify if the stored value is the same song id.

                            else:

                                if artists == user_exists[2]:

                                    new_list = user_exists[2]
                                
                                # If not, add it.

                                else:

                                    new_list = user_exists[2] + ',' + str(artists)

                            # Insert it to the db

                            query_6 = '''
                                    UPDATE users
                                    SET liked_artists = %s
                                    WHERE username = %s;
                                    '''
                            cursor.execute(query_6,(new_list,username))

                            print('User ', username,' succesfully added to the users table.')

                            connection.commit()

                    except (Exception, psycopg2.Error) as error:
                        print('Error adding the artist', artists, 'or the song', song_id, 'to the user ', username,'.')


        except (Exception, psycopg2.Error) as error:
                    print('Error verifying or creating the table.')

        finally:

            if (connection):
                cursor.close()
                connection.close()
                print('Connection closed.')

        

        # return 'Success'

if __name__ == "__main__":

    # login_and_store_songs(username='8ppupllf9815lb0jzdj4dot8x')