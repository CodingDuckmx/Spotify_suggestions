import os
import psycopg2

from dotenv import load_dotenv, find_dotenv
from pull_data import SpotBot

load_dotenv()

def find_similar_songs():

    ''' Pulls the songs, calculate their
        metric-based similar songs, and add
        them to a dictionary in the db.
    '''

    try:

        # Stablishes connection to the db.

        connection = psycopg2.connect(user = os.environ.get('db_user'),
                                    password = os.environ.get('db_password'),
                                    host = os.environ.get('db_host'),
                                    database = os.environ.get('db_name'))

        print('DB Connection:', connection)

        # Create the cursor.

        cursor = connection.cursor()
        
        print('DB Cursor:', cursor)

        cursor.execute('''
                    SELECT danceability, energy
                    FROM songs
        ''')

        songs = cursor.fetchall()

        if songs:

            print(songs)

        else:

            print('No songs fetched.')
    
    except (Exception, psycopg2.Error) as error:
        print('Error connecting or working with the database.')

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')

    breakpoint()

if __name__ == "__main__":
    
    find_similar_songs()