from flask import Blueprint, render_template, jsonify,request,flash, redirect, url_for
from flask_login import  login_required, current_user
from .models import import_ratings_csv
from website import db
from website.models import Movie, User, Rating, Link, Tag
from website.search import search_title

from website.keras import recommend_movies_keras
from website.knn import recommend_movies_knn,recommend_movies_knn_User

from website.cosine import recommend_movies_cosine,recommend_movies_cosine_User
from website.hybrid import recommend_movies_hybrid_User
from flask.globals import current_app
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_paginate import Pagination, get_page_parameter
import json



views = Blueprint('views',__name__)

@views.route('/')
@login_required
def home():
    return render_template("home.html", user=current_user)





@views.route('/rated_movies')
@login_required
def rated_movies():
    rated_movies = db.session.query(Movie, Rating,Link).join(Rating, Movie.id == Rating.movie_id).join(Link, Movie.id == Link.movie_id).filter(Rating.user_id == current_user.id).all()

    return render_template("rated_movies.html", rated_movies=rated_movies, user=current_user)



@views.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    user_ratings = Rating.query.filter_by(user_id=current_user.id).all()
    user_ratings = {r.movie_id: r.rating for r in user_ratings}
    if request.method == 'POST':
        title = request.form.get('title')
        with current_app.app_context():
            movie_df = search_title(title)
        return render_template('search.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,user_ratings=user_ratings,title=title, user=current_user)


        
    

@views.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    user_ratings = Rating.query.filter_by(user_id=current_user.id).all()
    user_ratings = {r.movie_id: r.rating for r in user_ratings}
    movieTitle = None
    
    if request.method == 'POST':
        title = request.form.get('title')
        recommend_type = request.form.get('recommend_type')
        recommend_type_button = request.form.get('recommend_type_button')
        if not title:
            if recommend_type_button == "KNNUser":
                movie_df, _, _ = recommend_movies_knn_User(user_id=current_user.id)
                return render_template('recommend.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,movieTitle=movieTitle,user_ratings=user_ratings,selectedRecommendSystem=recommend_type, user=current_user)
            elif recommend_type_button == "CosineUser":
                movie_df, _ = recommend_movies_cosine_User(user_id=current_user.id)
                return render_template('recommend.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,movieTitle=movieTitle,user_ratings=user_ratings,selectedRecommendSystem=recommend_type, user=current_user)
            elif recommend_type_button == "KerasUser":
                movie_df, _, _= recommend_movies_keras(user_id=current_user.id)
                return render_template('recommend.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,movieTitle=movieTitle,user_ratings=user_ratings,selectedRecommendSystem=recommend_type, user=current_user)
            elif recommend_type_button == "HybridUser":
                movie_df= recommend_movies_hybrid_User(user_id=current_user.id)
                return render_template('recommend.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,movieTitle=movieTitle,user_ratings=user_ratings,selectedRecommendSystem=recommend_type, user=current_user)
            
            flash("Please type in a movie title",category='error')
            return render_template('recommend.html',movieTitle=movieTitle,user_ratings=user_ratings)
        else:
            with current_app.app_context():
                if not recommend_type:
                    flash("Please select a recommendation algorithm.",category='error')
                    return render_template('recommend.html',movieTitle=movieTitle,user_ratings=user_ratings)
                
                elif recommend_type == 'Cosine':
                    movie_df,movieTitle, _ = recommend_movies_cosine(title)
                elif recommend_type == 'KNN':
                    movie_df,movieTitle, _ = recommend_movies_knn(title)
                
                
            return render_template('recommend.html',column_names=movie_df.columns.values, row_data=list(movie_df.values.tolist()),link_column="movie_id", zip=zip,movieTitle=movieTitle,user_ratings=user_ratings,selectedRecommendSystem=recommend_type, user=current_user)
    
    return render_template('recommend.html',movieTitle=movieTitle,user_ratings=user_ratings)
    


@views.route('/user', methods=['GET', 'POST'])
def user():
    if request.method == 'GET':
        all_users = User.query.all()
        return render_template('user.html', all_users=all_users)
    if request.method == 'POST':
        all_users = User.query.all()
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already used',category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.',category='error')   
        elif len(first_name) < 2:
            flash('First name must be greater than 1 characters.',category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.',category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.',category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')
            all_users = User.query.all()
            return render_template('user.html', all_users=all_users)
        
    return render_template("user.html", all_users=all_users)

@views.route('/update', methods=['POST', 'GET'])
def update():
    if request.method == 'POST':
        user_data = User.query.get(request.form.get('id'))
        if user_data:
            new_email = request.form['email']
            existing_user = User.query.filter_by(email=new_email).first()
            if existing_user and existing_user.id != user_data.id:
                flash('Email is already being used by another account', category='error')
            else:
                user_data.first_name = request.form['firstName']
                user_data.email = new_email
                db.session.commit()
        return redirect(url_for('views.user'))
    return redirect(url_for('views.user'))

@views.route('/delete/<int:user_id>', methods=['POST'])
def delete(user_id):
    user = User.query.get(user_id)
    if user:
        Rating.query.filter_by(user_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
    return redirect(url_for('views.user'))

@views.route('/movies', methods=['GET', 'POST'])
def movie():
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 100
    offset = (page - 1) * per_page
    

    if request.method == 'GET':
        
        movies = Movie.query.offset(offset).limit(per_page).all()
        total = Movie.query.count()
        pagination = Pagination(page=page, per_page=per_page, total=total, css_framework="bootstrap5")

        for movie in movies:
            links = Link.query.filter_by(movie_id=movie.id).all()
            movie.links = links

        return render_template('movies.html', all_movies=movies, pagination=pagination)
    
    if request.method == 'POST':

        movies = Movie.query.offset(offset).limit(per_page).all()
        total = Movie.query.count()

        for movie in movies:
            links = Link.query.filter_by(movie_id=movie.id).all()
            movie.links = links

        title = request.form.get('title')
        genreArray = json.loads(request.form.get('genreArray', '[]'))
        genre = "|".join(genreArray)
        
        movie = Movie.query.filter_by(title=title).first()
        if movie:
            flash('Movie already added',category='error')
        
        else:
            new_movie = Movie(title=title, genre=genre)
            db.session.add(new_movie)
            db.session.commit()
            flash('Movie added!', category='success')

            movies = Movie.query.offset(offset).limit(per_page).all()
            total = Movie.query.count()
            pagination = Pagination(page=page, per_page=per_page, total=total, css_framework="bootstrap5")

            for movie in movies:
                links = Link.query.filter_by(movie_id=movie.id).all()
                movie.links = links

        return render_template('movies.html', all_movies=movies, pagination=pagination)
        
    return render_template('movies.html', all_movies=movies, pagination=pagination)
    
@views.route('/deleteMovie/<int:movie_id>', methods=['POST'])
def deleteMovie(movie_id):
    movie = Movie.query.get(movie_id)
    if movie:
        Rating.query.filter_by(movie_id=movie_id).delete()
        Tag.query.filter_by(movie_id=movie_id).delete()
        db.session.delete(movie)
        db.session.commit()
    return redirect(request.referrer)

@views.route('/updateMovie', methods=['POST'])
def updateMovie():
    movie_id = request.form.get('id')
    updated_title = request.form.get('title')
    updated_genre = request.form.get('genre')

    movie = Movie.query.get(movie_id)
    if movie:
        movie.title = updated_title
        movie.genre = updated_genre
        db.session.commit()
        flash('Movie information updated!', category='success')
    else:
        flash('Movie not found.', category='error')

    return redirect(request.referrer)

@views.route('/update_rating', methods=['GET'])
@login_required
def update_rating():
    movie_id = request.args.get("movie_id")
    rating_value = request.args.get("rating_value")
    user_id = current_user.id
    existing_rating = Rating.query.filter_by(user_id=user_id, movie_id=movie_id).first()
    
    if existing_rating:
        # Update existing rating
        existing_rating.rating = rating_value
        existing_rating.timestamp = datetime.now().timestamp()
        db.session.commit()
        return jsonify(existing_rating.to_dict())
    else:
        # Create a new rating
        user = User.query.get(user_id)  
        movie = Movie.query.get(movie_id)
        rating = Rating(user=user, movie=movie, rating=rating_value, timestamp=datetime.now().timestamp())
        db.session.add(rating)
        db.session.commit()
        return jsonify(rating.to_dict())


        
