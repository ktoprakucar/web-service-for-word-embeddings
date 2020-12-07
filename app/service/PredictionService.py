import nltk
import numpy as np
import pandas as pd
from flask import current_app
from nltk.corpus import stopwords

from app.service.TextClassifierService import TextClassifierService
from app.service.WordEmbeddingsService import WordEmbeddingsService

wordEmbeddingsService = WordEmbeddingsService()
textClassifierService = TextClassifierService()


class PredictionService:

    def predict_text_class(self, text, model_id):
        current_app.logger.info("Text prediction started.")
        stop_words = stopwords.words('english')
        words = [x.lower() for x in nltk.word_tokenize(text) if x not in stop_words and x.isalnum()]
        word_vectors = wordEmbeddingsService.create_word_embeddings(words, model_id)
        comment_vector = pd.DataFrame(np.mean(word_vectors)).transpose()
        result = textClassifierService.predict(comment_vector, model_id)
        current_app.logger.info("Text prediction completed.")
        return result
