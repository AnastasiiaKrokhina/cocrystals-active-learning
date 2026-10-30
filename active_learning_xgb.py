import sys
import os
import pandas as pd
import numpy as np
import torch
import pickle
import time
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add the Streamlit app to path
sys.path.append(os.path.join(os.getcwd(), 'streamlitMolecularSetTransformer'))

# Import your Set Transformer Encoder and fingerprint generator
from deep_one_class.ecfp4 import PairsEncoder, get_representation

# --- 1. Feature Extractor (Using your trained Set Transformer) ---
class MSTFeatureExtractor:
    def __init__(self, model_path):
        self.encoder = PairsEncoder()
        model_dict = torch.load(model_path, map_location='cpu')
        
        # Get the autoencoder weights
        autoencoder_state_dict = model_dict['ae_net_dict']
        
        # CRITICAL FIX: Strip the 'encoder.' prefix from the keys
        # Since PairsAutoEncoder is built as: self.encoder = PairsEncoder()
        # The saved keys are "encoder.seq.0...", but PairsEncoder expects "seq.0..."
        stripped_state_dict = {}
        for key, value in autoencoder_state_dict.items():
            if key.startswith('encoder.'):
                # Remove the 'encoder.' prefix
                new_key = key[len('encoder.'):]
                stripped_state_dict[new_key] = value
            # Ignore 'decoder.' keys entirely (we don't need them)
        
        # Load the stripped weights into the bare encoder
        self.encoder.load_state_dict(stripped_state_dict)
        self.encoder.eval()  # Freeze dropout/batchnorm
        print("Loaded Set Transformer Encoder (stripped autoencoder weights).")

    def transform(self, smiles1_list, smiles2_list):
        """Convert SMILES lists to 50-dim embeddings."""
        if isinstance(smiles1_list, str):
            smiles1_list = [smiles1_list]
            smiles2_list = [smiles2_list]
        rep_df = get_representation(smiles1_list, smiles2_list)
        X = torch.FloatTensor(rep_df.values.astype(np.float32))
        with torch.no_grad():
            embeddings = self.encoder(X).numpy()  # (n_samples, 50)
        return embeddings

# --- 2. Active Learning Query Strategies ---
def query_by_uncertainty(probs, n_queries=20, bias='balanced'):
    """
    Select the most uncertain samples.
    
    Args:
        probs: Array of shape (n_samples, 2) -> [P(negative), P(positive)]
        n_queries: Number of samples to select.
        bias: 'balanced' (pick closest to 0.5), 'negative' (pick confident negatives but unsure).
    """
    p_pos = probs[:, 1]  # Probability of being positive (co-crystal)
    
    if bias == 'balanced':
        # Pure uncertainty: closest to 0.5
        uncertainty = np.abs(p_pos - 0.5)
    
    elif bias == 'negative':
        # We specifically want to FIND NEGATIVES (failures).
        # We target samples where the model is UNSURE but leans positive (p_pos ≈ 0.6)
        # OR samples where the model is confident they are negative (p_pos < 0.4) but has high variance.
        # However, standard "least confident" works too.
        # A clever trick: query samples where the model says "maybe positive, maybe not".
        # Let's use a score that heavily weights samples near 0.5.
        uncertainty = np.abs(p_pos - 0.5)
        # If we want to prioritize negatives, we can also give a bonus to samples predicted negative (p_pos < 0.5)
        # but still require some uncertainty.
        # For simplicity, we use pure uncertainty (Margin sampling) which is the gold standard.
        pass

    # Sort by uncertainty descending (most uncertain first)
    query_indices = np.argsort(uncertainty)[-n_queries:][::-1]
    return query_indices, uncertainty

def query_by_entropy(probs, n_queries=20):
    """Select samples with highest predictive entropy."""
    # Entropy = - sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    query_indices = np.argsort(entropy)[-n_queries:][::-1]
    return query_indices, entropy

# --- 3. Main Active Learning Class ---
class XGBActiveLearner:
    def __init__(self, feature_extractor, scale_pos_weight=None):
        self.feature_extractor = feature_extractor
        self.model = None
        self.scaler = None  # XGBoost doesn't require scaling, but it can help.
        self.scale_pos_weight = scale_pos_weight  # For imbalance (e.g., sum(neg)/sum(pos))
        
    def fit(self, X, y):
        """Train XGBoost on the current labeled set."""
        # Calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            neg_count = np.sum(y == 0)
            pos_count = np.sum(y == 1)
            if pos_count > 0:
                self.scale_pos_weight = neg_count / pos_count
            else:
                self.scale_pos_weight = 1.0
            print(f"  Scale_pos_weight: {self.scale_pos_weight:.2f}")

        # XGBoost parameters (tuned for molecular data)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': self.scale_pos_weight,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # Fast and memory efficient
            'random_state': 42,
            'verbosity': 0  # Silent training
        }
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y, verbose=False)
        print(f"  Model trained on {len(X)} samples.")
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def query(self, X_pool, n_queries=20, strategy='uncertainty'):
        """Query the most informative samples from the pool."""
        if self.model is None:
            raise ValueError("Model must be trained before querying.")
        
        probs = self.predict_proba(X_pool)  # (n_samples, 2)
        
        if strategy == 'uncertainty':
            query_idx, scores = query_by_uncertainty(probs, n_queries, bias='balanced')
        elif strategy == 'entropy':
            query_idx, scores = query_by_entropy(probs, n_queries)
        elif strategy == 'negative_biased':
            # We still use uncertainty but multiply by a factor to prefer negatives
            p_pos = probs[:, 1]
            # Uncertainty (closer to 0.5 is higher)
            uncertainty = np.abs(p_pos - 0.5)
            # Bias: if p_pos > 0.5 (predicted positive), we penalize it slightly
            # so we are more likely to query negatives (p_pos < 0.5) that are uncertain
            bias_factor = np.ones_like(p_pos)
            bias_factor[p_pos > 0.6] = 0.8  # Less likely to query clear positives
            bias_factor[p_pos < 0.4] = 1.2  # More likely to query predicted negatives
            score = uncertainty * bias_factor
            query_idx = np.argsort(score)[-n_queries:][::-1]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return query_idx, probs[query_idx], scores[query_idx]

# --- 4. The Active Learning Simulation Loop ---
def run_active_learning_simulation():
    # --- Load your data ---
    # CSV must have columns: 'smiles1', 'smiles2', 'label' (1=Positive, 0=Negative)
    data_path = 'data/cocrystals_ethanol.csv'  # CHANGE THIS
    df = pd.read_csv(data_path, sep='\t', header=0)
    print(f"Total samples: {len(df)}")
    print(f"  Positives: {sum(df['label']==1)}")
    print(f"  Negatives: {sum(df['label']==0)}")
    
    # --- Split: Hold-out test set (20%) ---
    test_df = df.sample(frac=0.2, random_state=42)
    pool_df = df.drop(test_df.index).reset_index(drop=True)
    
    # --- Initial training set (Start with only 10% of the data, but we need at least 5 negatives) ---
    # To simulate a real cold start, we take 80% of positives and 20% of negatives
    # But let's just take a random 15% of the pool as initial labeled set.
    initial_train = pool_df.sample(frac=0.15, random_state=42)
    pool_df = pool_df.drop(initial_train.index).reset_index(drop=True)
    
    print(f"\nInitial Train: {len(initial_train)} (Pos: {sum(initial_train['label']==1)}, Neg: {sum(initial_train['label']==0)})")
    print(f"Pool size: {len(pool_df)}")
    print(f"Test size: {len(test_df)}")
    
    # --- Initialize Feature Extractor (your trained MST) ---
    model_path = 'pretrained_models/full_deepsvdd_from_ecfp4.pth'
    feature_extractor = MSTFeatureExtractor(model_path)
    
    # --- Extract embeddings for the initial training set ---
    X_train = feature_extractor.transform(initial_train['smiles1'].tolist(), initial_train['smiles2'].tolist())
    y_train = initial_train['label'].values
    
    # --- Extract embeddings for the entire pool (to save time) ---
    print("\nExtracting pool embeddings... (This may take a moment)")
    X_pool = feature_extractor.transform(pool_df['smiles1'].tolist(), pool_df['smiles2'].tolist())
    y_pool_true = pool_df['label'].values  # We have them for simulation
    
    # --- Extract test embeddings ---
    X_test = feature_extractor.transform(test_df['smiles1'].tolist(), test_df['smiles2'].tolist())
    y_test = test_df['label'].values
    
    # --- Active Learning Loop ---
    n_rounds = 10
    queries_per_round = 20
    
    # For tracking
    auc_history = []
    
    learner = XGBActiveLearner(feature_extractor)
    
    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num+1}/{n_rounds} ---")
        start_time = time.time()
        
        # 1. Train XGBoost on current labeled set
        learner.fit(X_train, y_train)
        
        # 2. Evaluate on test set
        proba_test = learner.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba_test)
        auc_history.append(auc)
        print(f"  Test AUC: {auc:.4f}")
        
        # 3. Query the most uncertain samples from the pool
        # Strategy: 'uncertainty' (Margin), 'entropy', or 'negative_biased'
        query_idx, query_probs, query_scores = learner.query(
            X_pool, n_queries=queries_per_round, strategy='uncertainty'
        )
        
        # 4. SIMULATE LABELING: In real life, run experiments.
        # Here we just look up the true labels.
        query_labels = y_pool_true[query_idx]
        
        # 5. Add queried samples to the training set
        X_train = np.vstack([X_train, X_pool[query_idx]])
        y_train = np.concatenate([y_train, query_labels])
        
        # 6. Remove them from the pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool_true = np.delete(y_pool_true, query_idx, axis=0)
        pool_df = pool_df.drop(pool_df.index[query_idx]).reset_index(drop=True)
        
        # Log statistics
        n_neg = np.sum(query_labels == 0)
        n_pos = np.sum(query_labels == 1)
        print(f"  Queried {len(query_labels)} samples: {n_neg} Negatives, {n_pos} Positives")
        print(f"  Labeled set: {len(X_train)} (Neg: {np.sum(y_train==0)}, Pos: {np.sum(y_train==1)})")
        print(f"  Round time: {time.time() - start_time:.2f}s")
    
    # --- Final Evaluation ---
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    learner.fit(X_train, y_train)  # Final retrain on all labeled data
    proba_test = learner.predict_proba(X_test)[:, 1]
    final_auc = roc_auc_score(y_test, proba_test)
    print(f"Final Test AUC: {final_auc:.4f}")
    
    # Print classification report
    y_pred = learner.predict(X_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Neg (0)', 'Pos (1)']))
    
    # --- Save the final model ---
    with open('final_xgboost_model.pkl', 'wb') as f:
        pickle.dump(learner.model, f)
    print("\nFinal XGBoost model saved to 'final_xgboost_model.pkl'")
    
    # --- Plot Learning Curve ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_rounds+1), auc_history, marker='o', linestyle='-')
    plt.xlabel('Active Learning Round')
    plt.ylabel('Test AUC')
    plt.title('Active Learning Curve (XGBoost)')
    plt.grid(True)
    plt.savefig('active_learning_curve_xgboost.png', dpi=300)
    print("Learning curve saved to 'active_learning_curve_xgboost.png'")

if __name__ == "__main__":
    run_active_learning_simulation()