import csv
import os
import psycopg2

from dotenv import load_dotenv, find_dotenv

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

        query_000 = '''
                CREATE TABLE IF NOT EXISTS songs (
                    id SERIAL,
                    song_id VARCHAR(1024) UNIQUE NOT NULL PRIMARY KEY,
                    song_name VARCHAR(1024),
                    artists VARCHAR(1024),
                    album VARCHAR(1024),
                    popularity INT4,
                    explicit BOOL,
                    track_number INT4,
                    danceability FLOAT4,
                    energy FLOAT4,
                    song_key FLOAT4,
                    loudness FLOAT4,
                    song_mode INT4,
                    speechiness FLOAT4,
                    acousticness FLOAT4,
                    instrumentalness FLOAT4,
                    liveness FLOAT4,
                    valence FLOAT4,
                    tempo FLOAT4,
                    song_type TEXT,
                    duration_ms INT4,
                    time_signature INT4,
                    year INT4,
                    preview_url VARCHAR(2048),
                    similar_songs TEXT,
                    liked_by TEXT
                );
        '''

        cursor.execute(query_000)

        connection.commit()
        
        query_00 = '''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL,
                    username VARCHAR(2048) UNIQUE NOT NULL PRIMARY KEY,
                    liked_songs TEXT,
                    suggested_songs TEXT,
                    liked_artists TEXT,
                    suggested_artists TEXT,
                    following_playlists TEXT,
                    suggested_playlists TEXT
                );
        '''

        cursor.execute(query_00)

        connection.commit()
        
        query_0 = '''
                CREATE TABLE IF NOT EXISTS playlists (
                    id SERIAL,
                    playlist_id VARCHAR(2048) UNIQUE NOT NULL PRIMARY KEY,
                    listed_songs TEXT,
                    followed_by TEXT
                );
        '''

        cursor.execute(query_0)

        connection.commit()

        return 'Success'
    
    except (Exception, psycopg2.Error) as error:
        print(f'Error verifying or creating the table; {error}')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')


def store_dataset_songs():

    songs_dict = {}

    with open('.\Kaggle Datasets\data.csv','r',encoding='UTF-8',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for num, row in enumerate(spamreader):
            if num > 0:

                # dropping the release date, not relevant
                del row[-2]

                # adding the info to the dictionary
                if row[0] not in songs_dict:

                    songs_dict[row[0]] = row

    try:

        # Stablishes the connection to the db.

        connection = psycopg2.connect(user=os.environ.get('db_user'),
                                    password=os.environ.get('db_password'),
                                    host=os.environ.get('db_host'),
                                    port=os.environ.get('db_port'),
                                    database=os.environ.get('db_name'))

        print('DB Connection:', connection)

        cursor = connection.cursor()
        print('DB Cursor:', cursor)

    except (Exception, psycopg2.Error) as error:
        print(f'Error connection with the db. {error}')

    # Because this is a one time event,
    # we are not verifying the values we are up to
    # insert are already in the db.

    count = 1

    for key, value in songs_dict.items():

        try:

            query = '''
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
                    year
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
                    %s
                ); 
                '''

            cursor.execute(query, tuple(value))
            connection.commit()
            count += 1

        except:
            print(f'The song {key} was not succesfully added.')

    print(f'{count} songs were succesfully added. This is a {count / len(songs_dict)*100}% of total')

    if (connection):

        connection.close()
        print('Connection closed.')

if __name__ == "__main__":
       
    create_tables()
    store_dataset_songs()