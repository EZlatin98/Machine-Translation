#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pprint
import json
import re
import os
import pandas as pd
import csv
import string
import numpy as np

import pickle
import warnings
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

# Gensim
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel

# # spacy for lemmatization
# import nltk
# from nltk.corpus import stopwords
# import spacy
# # import en_core_web_sm

# # Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
# import matplotlib.pyplot as plt

# In[2]:

def city_county_map():
    out = {}
    with open('uscities.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for elem in reader:
            out[(elem["city"],elem['state_id'])] = elem["county_name"]

    return out


def find_county(line_prefix):
    *rest, state_id = line_prefix.split()
    return cty_cnty_map[(' '.join(rest), state_id)]


def find_state(line_prefix):
    *rest, state_id = line_prefix.split()
    return state_id


def get_week(file_name):
    if file_name.startswith("4-30"):
        return 2
    if file_name.startswith("4-21"):
        return 1
    return 0


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')



def clean_tweets():
    out_tweets = []


def format_tweet_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # Remove urls
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.replace('"', '')
    text = text.replace("'", '')

    words = text.split()
    # Remove the word RT, any words that don't contain ascii characters or contain digit characters
    words = list(filter(lambda x: x != "rt" and x.isascii() and not any(char.isdigit() for char in x), words))
    return words

def get_cleaned_tweets():
    tweets = []
    count = 0
    bad_format = 0
    bad_format_text = []
    files = ['test0.txt', 'test25.txt', 'test50.txt', 'test75.txt',
             'test100.txt', 'test125.txt', 'test150.txt', 'test175.txt',
             'test200.txt', 'test225.txt', 'test250.txt']
    # files = ['test0.txt']
    cty_cnty_map = city_county_map()

    good_format_files = [f'4-21test{25 * i}.txt' for i in range(11)] + [f'4-30test{25 * i}.txt' for i in range(11)]
    # good_format_files = []
    for file in files:
        with open(file, encoding="utf-8") as fp:
            line = fp.readline()
            while line:
                count += 1
                line = line.split(",{")
                county = find_county(line[0])
                state = find_state(line[0])
                line = "{" + line[1]
                line = deEmojify(line.replace("\'", "\""))
                line = line.replace("False", "\"False\"").replace("True", "\"True\"").replace("None", "\"None\"")
                line = line.replace("href=\"http:", "href='http:").replace("\\xa0", " ")
                line = re.sub('\"source.*?,', '', line)
                try:
                    y = json.loads(line)
                    y["county"] = county
                    y["state"] = state
                    y["date"] = get_week(file)
                    tweets.append(y)
                except:
                    bad_format += 1
                    bad_format_text.append(line)
                    pass
                line = fp.readline()

    for file in good_format_files:
        with open(file, encoding="utf-8") as fp:
            line = fp.readline()
            while line:
                start = line.find("{")
                county = find_county(line[0:start - 1])
                state = find_state(line[0:start - 1])
                line = line[start:]
                y = json.loads(line)
                y["county"] = county
                y["state"] = state
                y["date"] = get_week(file)
                tweets.append(y)
                line = fp.readline()

    county_pop = pd.read_csv("county_pop.csv")
    county_pop.head()

    # for i in range(20):
    #     pprint.pprint(tweets[i])

    cases_dict = {}
    with open('confirmed_covid.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for elem in reader:
            cases_dict[(elem['County Name'], elem['State'])] = elem['4/21/20']

    def get_confirmed_cases(tweet):
        cnty = tweet["county"]
        state = tweet["state"]
        pos_names = [cnty, cnty + " County", cnty + " Area"]
        for name in pos_names:
            if (name, state) in cases_dict:
                return cases_dict[(name, state)]

        assert False

    out_tweets = []
    for i, tweet in enumerate(tweets):
        cleaned_tweet = dict()
        if "retweeted_status" in tweet:
            tweet["full_text"] = tweet["retweeted_status"]["full_text"]
        if "quoted_status" in tweet:
            tweet["full_text"] += " " + tweet["quoted_status"]["full_text"]

        cleaned_tweet["text"] = format_tweet_text(tweet["full_text"])
        cleaned_tweet["date"] = tweet["date"]
        cleaned_tweet["favorite_count"] = tweet["favorite_count"]
        cleaned_tweet["is_retweet"] = "retweeted_status" in tweet
        cleaned_tweet["county"] = tweet["county"]
        cleaned_tweet["confirmed_cases"] = get_confirmed_cases(tweet)
        out_tweets.append(cleaned_tweet)
    return out_tweets


def write_to_file():
    f = open("file.pkl", "wb")
    pickle.dump(get_cleaned_tweets(), f)
    f.close()
    
def read_from_file():
    my_dict = pickle.load(open( "file.pkl", "rb"))
    return my_dict

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    
def preprocessLDA_sklearn(x, number_topics):
    y = pd.DataFrame(x)
    docs = [' '.join(word for word in y.loc[i]['text']) for i in range(len(y))]
    vec = CountVectorizer(docs, min_df=20, max_df=5000, stop_words='english')
    X = vec.fit_transform(docs)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
    
    warnings.simplefilter("ignore", DeprecationWarning)
    
    number_words = 10
    
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(X)
    
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, vec, number_words)
    
def preprocessLDA_gensim(postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    x = read_from_file()
    stop_words = stopwords.words('english') + ['amp']
    
    warnings.simplefilter("ignore", DeprecationWarning)
    
    y = pd.DataFrame(x)
    data_words = [' '.join(word for word in y.loc[i]['text']) for i in range(len(y))]
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # See trigram example
    # print(trigram_mod[bigram_mod[data_words[0]]])
    
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
        
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # nlp = en_core_web_sm.load()
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=postags)
    
    # print(data_lemmatized[:1])
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    
    # Create Corpus
    texts = data_lemmatized
    
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # View
    # print(corpus[:1])
    
    import pickle
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    id2word.save('id2word.gensim')
    
    return corpus, id2word
    
def buildLDA(corpus, id2word, num_topics, passes=10):
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=passes,
                                            alpha='auto',
                                            per_word_topics=True)
    pprint.pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    lda_model.save('model'+str(num_topics)+'.gensim')
    return lda_model
    
def run_topic_model(postags=['NOUN', 'ADJ'], num_topics=10, num_passes=15):
    
    print('preprocessing data...')
    corpus, id2word = preprocessLDA_gensim(['NOUN', 'ADJ'])
    
    print('modeling topics...')
    lda_model = buildLDA(corpus, id2word, num_topics, num_passes)
    
    return corpus, id2word, lda_model
    
def use_topic_model(num_topics=10):
    id2word = gensim.corpora.Dictionary.load('id2word.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model'+str(num_topics)+'.gensim')
    x = read_from_file()
    y = pd.DataFrame(x)
    freqs = [[cnt[1] for cnt in lda.get_document_topics(corpus[i])] for i in range(len(y))]
    
    f = open("freqs.pkl", "wb")
    pickle.dump(freqs, f)
    f.close()
    
    return freqs
    
def load_freqs():
    freqs = pickle.load(open( "freqs.pkl", "rb"))
    return freqs
    
#def printLDA(lda_model):
 #   # Print the Keyword in the 10 topics
  #  pprint.pprint(lda_model.print_topics())
   # doc_lda = lda_model[corpus]
    
    
    