import re
import pandas as pd

SUFFIX_MAP = {
    r'\bpvt\b': 'private',
    r'\bltd\b': 'limited',
    r'\binc\b': 'incorporated',
    r'\bcorp\b': 'corporation',
    r'\bco\b': 'company',
    r'\bllp\b': 'limited liability partnership',
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bste\b': 'suite',
}

def clean_text(text: str) -> str:
    """Standardizes string text: lowercase, expands abbreviations, strips punctuation."""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    for pattern, repl in SUFFIX_MAP.items():
        text = re.sub(pattern, repl, text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['clean_name'] = df['business_name'].apply(clean_text)
    df['clean_address'] = df['business_address'].apply(clean_text)
    df['clean_full'] = df['clean_name'] + " " + df['clean_address']
    # Country open-set support (US, India, France)
    df['country'] = df['country'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
    return df