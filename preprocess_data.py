# preprocess_data.py

import os
import re
import ast
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 0) — Download NLTK data (only the first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1) Load raw CMU metadata & plot summaries
metadata = pd.read_csv(
    'MovieSummaries/movie.metadata.tsv',
    sep='\t', header=None,
    names=['movie_id','mid','movie_name','release_date','movie_code',
           'rating','languages','countries','genres'],
    dtype={'movie_id':int}, encoding='utf-8'
)
plots = pd.read_csv(
    'MovieSummaries/plot_summaries.txt',
    sep='\t', header=None,
    names=['movie_id','plot_summary'],
    dtype={'movie_id':int,'plot_summary':str},
    encoding='utf-8'
)

# 2) Merge on movie_id
df = pd.merge(metadata, plots, on='movie_id', how='inner')
print(f"After merge: {len(df):,} records")

# 3) Clean the raw plot_summary text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # lower
    text = text.lower()
    # strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # strip Wiki templates {{…}}
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    # remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # keep only letters & spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords & single letters
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    # lemmatize
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmas)

print("Cleaning summaries (this may take a few minutes)…")
df['clean_summary'] = df['plot_summary'].apply(clean_text)

# 4) Extract genre names from Freebase‐style JSON string
def extract_genres(gen_str):
    try:
        d = ast.literal_eval(gen_str)
        return list(d.values())
    except Exception:
        return []

df['cleaned_genres'] = df['genres'].apply(extract_genres)

# 5) Build the full genre set & binary vectors
all_genres = sorted({g for lst in df['cleaned_genres'] for g in lst})
print(f"Detected {len(all_genres)} unique genres")

def to_binary(gen_list):
    return [1 if g in gen_list else 0 for g in all_genres]

df['genre_labels'] = df['cleaned_genres'].apply(to_binary)

# 6) Preview & save
print(df[['movie_id','clean_summary','cleaned_genres','genre_labels']].head())
out = df[['movie_id','clean_summary','cleaned_genres','genre_labels']]
out.to_csv('processed_cleaned_data.csv', index=False)
print("Wrote ➞ processed_cleaned_data.csv")
