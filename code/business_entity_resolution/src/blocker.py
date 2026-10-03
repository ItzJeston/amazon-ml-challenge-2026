import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class HybridBlocker:
    """Generates candidate pairs for each S1 entity using TF-IDF n-grams grouped by country."""
    def __init__(self, top_k=20):
        self.top_k = top_k
        
    def fit_predict(self, df_s1: pd.DataFrame, df_candidates: pd.DataFrame):
        candidates_dict = {s1_id: set() for s1_id in df_s1['entity_id']}
        all_countries = set(df_s1['country'].unique()).union(set(df_candidates['country'].unique()))
        
        for country in tqdm(all_countries, desc="Blocking by Country"):
            sub_s1 = df_s1[df_s1['country'] == country]
            sub_cand = df_candidates[df_candidates['country'] == country]
            
            if sub_s1.empty:
                continue
            if sub_cand.empty:
                sub_cand = df_candidates # Fallback if country not present in candidates
                
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
            cand_vectors = vectorizer.fit_transform(sub_cand['clean_full'])
            s1_vectors = vectorizer.transform(sub_s1['clean_full'])
            
            k = min(self.top_k, cand_vectors.shape[0])
            nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
            nn.fit(cand_vectors)
            
            distances, indices = nn.kneighbors(s1_vectors)
            
            s1_ids = sub_s1['entity_id'].values
            cand_ids = sub_cand['entity_id'].values
            
            for idx, s1_id in enumerate(s1_ids):
                candidates_dict[s1_id].update(cand_ids[indices[idx]])
                
        return candidates_dict