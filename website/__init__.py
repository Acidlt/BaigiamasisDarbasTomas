from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'TomasZuromskasITVF19'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    data_imported = True
    @app.before_first_request
    def import_data():
        nonlocal data_imported
        if not data_imported:
            import_ratings_csv()
            import_movies_csv()
            import_links_csv()
            import_tags_csv()
            data_imported = True
            print('Data imported successfully.')
        else:
            print('Data already imported.')

    

    from .models import User, Rating, Link,Tag, import_ratings_csv, import_movies_csv, import_links_csv, import_tags_csv

    with app.app_context():
        db.create_all()
        import_data()
    
    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')


    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))
    
   
    
    
    
    


   

    return app


