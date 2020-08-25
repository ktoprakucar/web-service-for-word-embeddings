class ModelVo:
    def __init__(self):
        self.model_id = None
        self.auc_score = None
        self.f_score = None

    def set_model_id(self, model_id):
        self.model_id = model_id

    def set_auc_score(self, auc_score):
        self.auc_score = auc_score

    def set_f_score(self, f_score):
        self.f_score = f_score
