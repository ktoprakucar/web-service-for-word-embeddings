# web-service-for-word-embeddings
A basic web service which can train a language model and an ensemble model to predict text classes.

Language model = Gensim

Ensemble model = Light Gradient Boosting Model

Web Service Framework = Flask

There are 3 functions in the Web Service. You can find the name of each function and their example cURL scripts:

* train word2vec model

curl POST 'http://localhost:8080/api/v1/wv-model-training' \
--form 'file={file_location}'


* train LGBM

curl POST http://localhost:8080/api/v1/text-classifier-training \
--form 'file={file_location}' \
--form 'model_id="{model_id_which_is_returned_from_the_first_method}"'

* predict text class

curl POST http://localhost:8080/api/v1/predict-text \
--header "Content-Type: application/json" \
--data '{"text": "{any_review_you_want_to_classify}",
"model_id": "{model_id_which_is_returned_from_the_first_method}"}'