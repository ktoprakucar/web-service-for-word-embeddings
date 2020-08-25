import json

import pandas as pd
from flask import Blueprint, current_app, request, jsonify

from app.service.TextClassifierService import TextClassifierService
from app.service.WordEmbeddingsService import WordEmbeddingsService

controller = Blueprint("classifier-controller", __name__)

wordEmbeddingsService = WordEmbeddingsService()
textClassifierService = TextClassifierService()


@controller.route('/wv-model-training', methods=['POST'])
def train_wv_model():
    current_app.logger.info("Model training started.")
    csv_file = request.form['file']
    with open(csv_file, encoding="utf8") as file:
        text_df = pd.read_csv(file)
    model_id = wordEmbeddingsService.train_model(text_df)
    return jsonify(model_id=model_id)


@controller.route('/text-classifier-training/<model_id>', methods=['POST'])
def train_classifier_model(model_id):
    current_app.logger.info("Text classifier model training started.")
    csv_file = request.form['file']
    with open(csv_file, encoding="utf8") as file:
        text_df = pd.read_csv(file)
    model_info = textClassifierService.train_classifier_model(text_df, model_id)
    return jsonify(json.dumps(model_info.__dict__))
