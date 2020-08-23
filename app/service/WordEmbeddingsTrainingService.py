import datetime

import gensim
import nltk
from flask import current_app
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('stopwords')


class WordEmbeddingsTrainingService:

    def train_model(self, text_df):
        tokenized_text = self.clean_and_tokenize_text(text_df)
        current_app.logger.info("Word2Vec model training started.")
        model = gensim.models.Word2Vec(tokenized_text, window=50,
                                       size=150,
                                       iter=5,
                                       min_count=3,
                                       workers=4)
        current_app.logger.info("Word2Vec model training completed.")
        model_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_model(model, model_id)
        return model_id

    def clean_and_tokenize_text(self, text_df):
        stop_words = stopwords.words('english')
        text_list = [[x for x in nltk.word_tokenize(x) if x not in stop_words and x.isalnum()] for x in text_df['Text']]
        return text_list

    def save_model(self, model, model_id):
        model_path = "app/models/wv/" + model_id + ".model"
        model.save(model_path)
        current_app.logger.info("Word2Vec model was saved: %s", model_id)
