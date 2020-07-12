import logging

from flask import Flask

from app.controller.WordEmbeddingsController import controller

logging.basicConfig(level=logging.INFO)


def create_app():
    application = Flask(__name__)
    application.register_blueprint(controller, url_prefix='/api/v1/word-embedding')
    return application
