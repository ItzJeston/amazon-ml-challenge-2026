import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from tqdm import tqdm

def extract_features(df_pairs: pd.DataFrame) -> np.ndarray:
    features = []
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Extracting Similarity Features"):
        n1, n2 = row['clean_name_s1'], row['clean_name_s2']
        a1, a2 = row['clean_address_s1'], row['clean_address_s2']
        
        features.append([
            fuzz.ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            distance.JaroWinkler.similarity(n1, n2),
            fuzz.ratio(a1, a2) / 100.0,
            fuzz.token_sort_ratio(a1, a2) / 100.0,
            distance.JaroWinkler.similarity(a1, a2),
            fuzz.ratio(row['clean_full_s1'], row['clean_full_s2']) / 100.0,
            abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
            abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
        ])
    return np.array(features)

def train_lgbm(X: np.ndarray, y: np.ndarray):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model.fit(X, y)
    return model