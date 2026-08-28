import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import scipy.stats
from models import ASAGCrossEncoder

def compute_qwk(y_true, y_pred, min_rating, max_rating):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def evaluate_model(model_dir, dataset_path, max_score=10):
    print(f"Loading model from {model_dir}...")
    model = ASAGCrossEncoder(model_name=model_dir)
    
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    if not all(col in df.columns for col in ['reference_answer', 'student_answer', 'score']):
        raise ValueError("Dataset must contain 'reference_answer', 'student_answer', and 'score' columns.")
        
    df = df.dropna(subset=['reference_answer', 'student_answer', 'score'])
    
    references = df['reference_answer'].astype(str).tolist()
    students = df['student_answer'].astype(str).tolist()
    true_scores = df['score'].values
    
    print("Generating predictions...")
    preds_normalized = model.predict(references, students)
    preds_scaled = np.array(preds_normalized) * max_score
    
    # Calculate metrics
    mse = mean_squared_error(true_scores, preds_scaled)
    qwk = compute_qwk(true_scores, preds_scaled, 0, max_score)
    pcc, _ = scipy.stats.pearsonr(preds_scaled, true_scores)
    
    print("\n=== Evaluation Results ===")
    print(f"MSE: {mse:.4f}")
    print(f"QWK (Quadratic Weighted Kappa): {qwk:.4f}")
    print(f"Pearson Correlation: {pcc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASAG Model")
    parser.add_argument("--model_dir", type=str, default="../cross_encoder_best")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--max_score", type=int, default=10)
    args = parser.parse_args()
    
    evaluate_model(args.model_dir, args.dataset, args.max_score)
