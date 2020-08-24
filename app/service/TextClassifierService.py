import nltk
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download('stopwords')

from flask import current_app
from app.service.WordEmbeddingsService import WordEmbeddingsService

wordEmbeddingsService = WordEmbeddingsService()


class TextClassifierService:

    def train_model(self, text_df, model_id):
        current_app.logger.info("LGBM model training started.")
        tokenized_text = self.clean_and_tokenize_text(text_df)
        wordEmbeddingsService.create_word_embeddings(tokenized_text, model_id)

    def clean_and_tokenize_text(self, text_df):
        stop_words = stopwords.words('english')
        text_list = [[x.lower() for x in nltk.word_tokenize(x) if x not in stop_words and x.isalnum()]
                     for x in text_df['Text']]
        return text_list
