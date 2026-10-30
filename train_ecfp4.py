# import sys
# import os
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset

# # Add the Streamlit app to path
# sys.path.append(os.path.join(os.getcwd(), 'streamlitMolecularSetTransformer'))

# # Import the DeepSVDD class
# from deep_one_class.src.deepSVDD import DeepSVDD
# from deep_one_class.src.base.base_dataset import BaseADDataset

# # CRITICAL: Import the build functions from ecfp4.py
# # This overrides the ones inside deepSVDD.py
# from deep_one_class.ecfp4 import build_network, build_autoencoder, PairsEncoder, PairsAutoEncoder

# # --- But wait! ecfp4.py already does this internally! ---
# # In ecfp4.py, it has:
# #   deepSVDD.build_network = build_network
# #   deepSVDD.build_autoencoder = build_autoencoder
# # So as long as we import ecfp4, the overrides are applied.
# # However, we need to import ecfp4 BEFORE creating DeepSVDD.

# # Actually, let's just call the override manually to be safe:
# import deep_one_class.src.deepSVDD as deepSVDD_module
# deepSVDD_module.build_network = build_network
# deepSVDD_module.build_autoencoder = build_autoencoder

# # --- 1. Dataset (reuse from ecfp4's get_representation) ---
# from deep_one_class.ecfp4 import get_representation, fingerprint_from_df, smile_to_fingerprint

# class CoCrystalDataset(Dataset):
#     def __init__(self, df, smiles1_col='smiles1', smiles2_col='smiles2'):
#         # Use ecfp4's get_representation to generate fingerprints
#         rep_df = get_representation(df[smiles1_col].values, df[smiles2_col].values)
#         self.data = rep_df.values.astype(np.float32)
#         self.labels = np.zeros(len(self.data), dtype=np.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# class MyADDataset(BaseADDataset):
#     def __init__(self, df, train_ratio=0.8, smiles1_col='smiles1', smiles2_col='smiles2'):
#         super().__init__()
#         train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42)
#         self.train_set = CoCrystalDataset(train_df, smiles1_col, smiles2_col)
#         self.test_set = CoCrystalDataset(test_df, smiles1_col, smiles2_col)
#         self.n_features = 8192

# # --- 2. Main Training ---
# def main():
#     # Load your CSV
#     df = pd.read_csv('/home/external/krokhina_a_e/co-crystals/MolecularSetTransformer/csd_data/csd_training_set.csv')
#     # Rename columns if needed (adjust to your CSV's column names)
#     # df = df.rename(columns={'SMILES1': 'smiles1', 'SMILES2': 'smiles2'})
    
#     dataset = MyADDataset(df, train_ratio=0.8, smiles1_col='smiles1', smiles2_col='smiles2')
#     print(f"Train samples: {len(dataset.train_set)}, Test samples: {len(dataset.test_set)}")

#     # --- Initialize DeepSVDD ---
#     # The architecture will come from ecfp4.py because we overrode the build functions
#     deep_svdd = DeepSVDD(objective='one-class', nu=0.05)
#     deep_svdd.net_name = 'fingerprint'  # Dummy name, build_network uses it
    
#     # Manually set the networks using ecfp4's builders
#     deep_svdd.net = build_network('fingerprint')
#     deep_svdd.ae_net = build_autoencoder('fingerprint')

#     # --- Stage 1: Pretrain ---
#     print("\n>>> Stage 1: Pretraining Autoencoder...")
#     deep_svdd.pretrain(
#         dataset=dataset,
#         optimizer_name='adam',
#         lr=0.001,
#         n_epochs=100,
#         batch_size=64,
#         weight_decay=1e-5,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )

#     # --- Stage 2: Deep SVDD ---
#     print("\n>>> Stage 2: Training Deep SVDD...")
#     deep_svdd.train(
#         dataset=dataset,
#         optimizer_name='adam',
#         lr=0.001,
#         n_epochs=50,
#         batch_size=64,
#         weight_decay=1e-5,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )

#     # --- Save Model ---
#     os.makedirs('pretrained_models', exist_ok=True)
#     model_path = 'pretrained_models/full_deepsvdd_from_ecfp4.pth'
#     deep_svdd.save_model(model_path, save_ae=True)
#     print(f"\n>>> Model saved to {model_path}")
#     print(f"R = {deep_svdd.R:.4f}")
#     print(f"c (first 5): {deep_svdd.c[:5]}")

# if __name__ == "__main__":
#     main()


import sys
import os
import pandas as pd
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Add the Streamlit app to path
sys.path.append(os.path.join(os.getcwd(), 'streamlitMolecularSetTransformer'))

# Import the DeepSVDD class
from deep_one_class.src.deepSVDD import DeepSVDD
from deep_one_class.src.base.base_dataset import BaseADDataset

# CRITICAL: Import the build functions from ecfp4.py
# This overrides the ones inside deepSVDD.py
from deep_one_class.ecfp4 import build_network, build_autoencoder, get_representation

# Apply the overrides manually to the deepSVDD module
import deep_one_class.src.deepSVDD as deepSVDD_module
deepSVDD_module.build_network = build_network
deepSVDD_module.build_autoencoder = build_autoencoder

# --- 1. Dataset (reuse from ecfp4's get_representation) ---
class CoCrystalDataset(Dataset):
    def __init__(self, df, smiles1_col='smiles1', smiles2_col='smiles2'):
        # Use ecfp4's get_representation to generate fingerprints
        rep_df = get_representation(df[smiles1_col].values, df[smiles2_col].values)
        self.data = rep_df.values.astype(np.float32)
        self.labels = np.zeros(len(self.data), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx]), idx

class MyADDataset(BaseADDataset):
    """Wraps the fingerprint dataset into the BaseADDataset format."""
    def __init__(self, df, root='', train_ratio=0.8, smiles1_col='smiles1', smiles2_col='smiles2'):
        super().__init__(root=root)   # <-- Pass root argument to base class
        # Split into train and test DataFrames
        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42)
        
        self.train_set = CoCrystalDataset(train_df, smiles1_col, smiles2_col)
        self.test_set = CoCrystalDataset(test_df, smiles1_col, smiles2_col)
        self.n_features = 8192  # Two 4096-bit fingerprints

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0):
        train_loader = DataLoader(
            self.train_set, 
            batch_size=batch_size, 
            shuffle=shuffle_train, 
            num_workers=num_workers
        )
        test_loader = DataLoader(
            self.test_set, 
            batch_size=batch_size, 
            shuffle=shuffle_test, 
            num_workers=num_workers
        )
        return train_loader, test_loader

# --- 2. Main Training ---
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Load your CSV
    df = pd.read_csv('/home/external/krokhina_a_e/co-crystals/MolecularSetTransformer/csd_data/csd_training_set.csv')
    # Rename columns if needed (adjust to your CSV's column names)
    # df = df.rename(columns={'SMILES1': 'smiles1', 'SMILES2': 'smiles2'})
    
    dataset = MyADDataset(df, train_ratio=0.8, smiles1_col='smiles1', smiles2_col='smiles2')
    print(f"Train samples: {len(dataset.train_set)}, Test samples: {len(dataset.test_set)}")

    # --- Initialize DeepSVDD ---
    # The architecture will come from ecfp4.py because we overrode the build functions
    deep_svdd = DeepSVDD(objective='one-class', nu=0.05)
    deep_svdd.net_name = 'fingerprint'  # Dummy name, build_network uses it
    
    # Manually set the networks using ecfp4's builders
    deep_svdd.net = build_network('fingerprint')
    deep_svdd.ae_net = build_autoencoder('fingerprint')

    # --- Stage 1: Pretrain ---
    print("\n>>> Stage 1: Pretraining Autoencoder...")
    deep_svdd.pretrain(
        dataset=dataset,
        optimizer_name='adam',
        lr=0.001,
        n_epochs=100,
        batch_size=64,
        weight_decay=1e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # --- Stage 2: Deep SVDD ---
    print("\n>>> Stage 2: Training Deep SVDD...")
    deep_svdd.train(
        dataset=dataset,
        optimizer_name='adam',
        lr=0.0001,
        n_epochs=50,
        batch_size=64,
        weight_decay=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # --- Save Model ---
    os.makedirs('pretrained_models', exist_ok=True)
    model_path = 'pretrained_models/full_deepsvdd_from_ecfp4.pth'
    deep_svdd.save_model(model_path, save_ae=True)
    print(f"\n>>> Model successfully saved to {model_path}")
    print(f"R = {deep_svdd.R:.4f}")
    print(f"c (first 5): {deep_svdd.c[:5]}")

    # --- Optional: Test and print AUC ---
    print("\n>>> Evaluating on test set...")
    deep_svdd.test(dataset, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test AUC: {deep_svdd.results['test_auc']:.4f}")

if __name__ == "__main__":
    main()
