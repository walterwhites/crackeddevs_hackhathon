###############################################################
# Author: walterwhites
# Hackathon: CrackedDevs Hackathon Jan 2024
# This code is subject Devpost Hackathon and restrictions.
###############################################################

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import sent_tokenize, WordNetLemmatizer
import re
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

def search(query, model, df_cleaned):
    query = preprocess_text(query)
    query_vector = np.mean([model.wv[word] for word in query.split() if word in model.wv], axis=0)

    description_vectors = [np.mean([model.wv[word] for word in desc.split() if word in model.wv], axis=0) for desc in df_cleaned['cleaned_description']]

    cosine_similarities = [linear_kernel([query_vector], [desc_vector]).flatten()[0] for desc_vector in description_vectors]

    results = pd.DataFrame({'id': df_cleaned['id'], 'Description': df_cleaned['cleaned_description'], 'Similarity': cosine_similarities})
    results = results.sort_values(by='Similarity', ascending=False)

    return results

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_urls(text)
    # Tokenization
    word_tokens = tokenization(text)
    word_tokens = clear_ponctuation(word_tokens)
    word_tokens = custom_clean(word_tokens)
    # Lemmatisation
    lemmatized_words = lemmatization(word_tokens)
    # Suppression des stopwords
    cleaned_text = remove_stopwords(lemmatized_words)
    cleaned_text_str = ' '.join(cleaned_text)  # Convert the list to a string
    return cleaned_text_str

def remove_urls(text):
    regex = r'https?://\S+|www\.\S+'
    text = re.sub(regex, '', text)
    return text

def lowercase_text(text):
    return text.lower()

def lemmatization(word_tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in word_tokens]

def tokenization(text):
    return sent_tokenize(text)

def custom_clean(text):
    # Remplacer les sauts de lignes par des espaces
    word_tokens = [word_token.replace('\n', ' ') for word_token in text]
    # Remplacer les non-breaking space (nbsp) par des espaces
    word_tokens = [re.sub(r'\xa0', ' ', phrase) for phrase in word_tokens]
    # Supprimer les espaces en trop
    word_tokens = [re.sub(r'\s+', ' ', phrase) for phrase in word_tokens]
    return word_tokens

def remove_stopwords(sentences):
    stop_words = set(stopwords.words('english'))
    result = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        result.append(' '.join(filtered_words))
    return result

def clear_ponctuation(text):
    return [re.sub(r'[^a-zA-Z0-9\s]', '', phrase) for phrase in text]

def extract_text_from_body(html_body):
    soup = BeautifulSoup(html_body, 'html.parser')
    return soup.get_text()