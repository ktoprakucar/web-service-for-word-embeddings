from flask import Blueprint, current_app, request, jsonify
import pandas as pd

from app.service.WordEmbeddingsTrainingService import WordEmbeddingsTrainingService

controller = Blueprint("credit-score", __name__)

wordEmbeddingsTrainingService = WordEmbeddingsTrainingService()


@controller.route('/wv-model-training', methods=['POST'])
def train_wv_model():
    current_app.logger.info("Model training started.")
    csv_file = request.form['file']
    with open(csv_file, encoding="utf8") as file:
        text_df = pd.read_csv(file)
    model_id = wordEmbeddingsTrainingService.train_model(text_df)
    return jsonify(model_id=model_id)
