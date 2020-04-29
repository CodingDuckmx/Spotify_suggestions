import urllib.parse as up

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():

    app = Flask(__name__,instance_relative_config=False) #the configuration is comming from other file.
    app.config.from_object('config.Config')

    db.init_app(app)

    with app.app_context():

        # Import routes

        from . import routes

        db.create_all()
        
        return app