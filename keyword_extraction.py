import itertools
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.nlp import max_sum_sim, mmr

df = pd.read_csv('ds_scraped_lists.csv')

keyword_list_max_sum = []
keyword_list_mmr = []
for i in tqdm(df.index):
    tmp_desc = df.loc[i, 'description']

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words='english'
                            ).fit([tmp_desc])
    candidates = count.get_feature_names()

    # bert
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([tmp_desc])
    candidate_embeddings = model.encode(candidates)

    top_n = 10
    # distances = cosine_similarity(doc_embedding, candidate_embeddings)
    # keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    # nr candidate to be tuned
    keywords_max_sum_sim = max_sum_sim(doc_embedding, candidate_embeddings,
                                       candidates, top_n=top_n,
                                       nr_candidates=20)
    keywords_mmr = mmr(doc_embedding, candidate_embeddings, candidates,
                       top_n=top_n,
                       diversity=0.7)

    keyword_list_max_sum.append(keywords_max_sum_sim)
    keyword_list_mmr.append(keywords_mmr)

df['keywords_max_sum'] = keyword_list_max_sum
df['keywords_mmr'] = keyword_list_mmr

# evaluate qualifications
df = df.drop_duplicates(
    subset=['city', 'company_name', 'category', 'job_title', 'qualified',
            'tech_compatible', 'major_compatible', 'app_link'])
df = df.sort_values(
    by=['qualified', 'tech_compatible', 'major_compatible', 'city',
        'company_name', 'category'], ascending=[0, 0, 0, 0, 0, 1])
df.to_csv('output.csv', encoding='utf-8', index=False)

# recommend jobs based on user's rating (outcome)
# as they provide more rating, the model performance will improve
# input features can be a variety of factors

"""
# design for personalization method over time

OUTPUT:
user rating (likert scale 1-7)

INPUT:
description-char
- keywords
- sentiment of the job description

job-char
- locations
- industry

company-char
- current rating of the company on glassdoor/indeed
- match w/ the candidate's current profile
"""

# todo covert keywords (or their embeddings) as features

# todo predict relationship between keywords and user's listing preference
