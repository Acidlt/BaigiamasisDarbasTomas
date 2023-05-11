from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func
import csv
import pandas as pd
from tqdm import tqdm

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    ratings = db.relationship('Rating', backref='user', lazy=True)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    genre = db.Column(db.String(200), nullable=False)

class Rating(db.Model):
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    rating = db.Column(db.Float)
    timestamp = db.Column(db.Integer)
    movie = db.relationship('Movie', backref='ratings', lazy=True)
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'movie_id': self.movie_id,
            'rating': self.rating,
            'timestamp': self.timestamp
        }
    
class Link(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    imdb_id = db.Column(db.String(20))
    tmdb_id = db.Column(db.String(20))   

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    tag = db.Column(db.String(200), nullable=False)
    movie = db.relationship("Movie", backref="tags")
    user = db.relationship("User", backref="tags")

def import_ratings_csv():
    ratings_df = pd.read_csv(r'C:\Users\ttzur\Desktop\WebWithFlask\website\static\ratings.csv')
    for index, row in tqdm(ratings_df.iterrows(), total=len(ratings_df), desc="Importing ratings"):
        rating = Rating(
            user_id=int(row['userId']),
            movie_id=int(row['movieId']),
            rating=float(row['rating']),
            timestamp=int(row['timestamp'])
        )
        db.session.add(rating)
        if index % 1000000 == 0:  
            db.session.commit()
    db.session.commit()
def import_movies_csv():
    
    with open(r'C:\Users\ttzur\Desktop\WebWithFlask\website\static\movies.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            movie = Movie(
                id=int(row[0]),
                title=row[1],
                genre=row[2],
            )
            db.session.add(movie)
    db.session.commit()
def import_links_csv():
    with open(r'C:\Users\ttzur\Desktop\WebWithFlask\website\static\links.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            link = Link(
                movie_id=int(row[0]),
                imdb_id=row[1] if row[1] != "" else None,
                tmdb_id=row[2] if row[2] != "" else None,
            )
            db.session.add(link)
    db.session.commit()

def import_tags_csv():
    with open(r'C:\Users\ttzur\Desktop\WebWithFlask\website\static\tags.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            tag = Tag(
                user_id=int(row[0]),
                movie_id=int(row[1]),
                tag=row[2],
            )
            db.session.add(tag)
    db.session.commit()
    