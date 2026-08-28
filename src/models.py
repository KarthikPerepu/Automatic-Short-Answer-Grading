import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import xgboost as xgb
import numpy as np

class ASAGCrossEncoder:
    def __init__(self, model_name="microsoft/deberta-v3-small", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.to(self.device)
        
    def predict(self, reference, student):
        """
        Predicts the normalized score (0 to 1) given a reference and student answer.
        """
        self.model.eval()
        # Handle batch or single pair
        is_single = isinstance(reference, str)
        if is_single:
            reference = [reference]
            student = [student]
            
        inputs = self.tokenizer(reference, student, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(-1)
            
        scores = torch.clamp(logits, 0.0, 1.0).cpu().numpy()
        return scores[0] if is_single else scores.tolist()

class ASAGMetaEnsemble:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
        self.is_trained = False
        
    def extract_features(self, cross_encoder_scores, student_answers):
        features = []
        for ce_score, ans in zip(cross_encoder_scores, student_answers):
            length = len(str(ans).split())
            features.append([ce_score, length])
        return np.array(features)
        
    def train(self, cross_encoder_scores, student_answers, true_scores):
        X = self.extract_features(cross_encoder_scores, student_answers)
        self.model.fit(X, true_scores)
        self.is_trained = True
        
    def predict(self, cross_encoder_scores, student_answers):
        if not self.is_trained:
            raise ValueError("Ensemble model is not trained yet.")
        X = self.extract_features(cross_encoder_scores, student_answers)
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.0)
