from flask import Flask
from .main import main

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    from .views import views
    app.register_blueprint(views)
    return app
