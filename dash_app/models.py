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