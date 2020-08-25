import datetime
import pickle

import nltk
import numpy as np
from flask import current_app
from lightgbm import *
from nltk.corpus import stopwords
from sklearn import *
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from app.service.WordEmbeddingsService import WordEmbeddingsService
from app.vo.ModelInfoVo import ModelVo

nltk.download("punkt")
nltk.download('stopwords')

wordEmbeddingsService = WordEmbeddingsService()


class TextClassifierService:

    def train_classifier_model(self, text_df, model_id):
        current_app.logger.info("LGBM model training started.")
        tokenized_text = self.clean_and_tokenize_text(text_df)
        predicates = wordEmbeddingsService.create_word_embeddings(tokenized_text, model_id)
        x_test, x_train, y_test, y_train = self.create_model_data(predicates, text_df)
        classifier = LGBMClassifier(boosting_type='gbdt',
                                    metric='binary_loglass',
                                    max_depth=7,
                                    n_jobs=4,
                                    n_estimators=80,
                                    reg_alpha=0.3,
                                    reg_lambda=0.8,
                                    is_unbalance=True,
                                    random_state=12345)
        classifier.fit(x_train, y_train)
        current_app.logger.info("LGBM model training completed.")
        predictions = classifier.predict(x_test)
        auc_score = roc_auc_score(y_test, predictions)
        f_score = f1_score(y_test, predictions, average='weighted')
        model_info = self.create_model_info(auc_score, f_score)
        self.save_model(classifier, model_id)
        current_app.logger.info('Models is saved: ' + model_info.model_id)
        return model_info

    def create_model_data(self, predicates, text_df):
        target = np.where(text_df['Score'] > 3, 1, 0)
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(target)
        x_train, x_test, y_train, y_test = train_test_split(predicates, target,
                                                            stratify=target,
                                                            test_size=0.3)
        return x_test, x_train, y_test, y_train

    def create_model_info(self, auc_score, f_score):
        model_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        model_info = ModelVo()
        model_info.set_model_id(model_id)
        model_info.set_auc_score(auc_score)
        model_info.set_f_score(f_score)
        return model_info

    def save_model(self, classifier, model_id):
        model_path = "app/models/classifier/" + model_id + ".model"
        pickle.dump(classifier, open(model_path, "wb"))

    def clean_and_tokenize_text(self, text_df):
        stop_words = stopwords.words('english')
        text_list = [[x.lower() for x in nltk.word_tokenize(x) if x not in stop_words and x.isalnum()]
                     for x in text_df['Text']]
        return text_list
