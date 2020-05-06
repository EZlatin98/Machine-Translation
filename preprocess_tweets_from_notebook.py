#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pprint
import json
import re
import pandas as pd
import csv
import string
import numpy as np


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

tweets = []
count = 0
bad_format = 0
bad_format_text = []
files = ['test0.txt', 'test25.txt', 'test50.txt','test75.txt',
         'test100.txt','test125.txt', 'test150.txt','test175.txt',
         'test200.txt','test225.txt','test250.txt']
# files = ['test0.txt']
cty_cnty_map = city_county_map()

good_format_files = [f'4-21test{25*i}.txt' for i in range(11)] + [f'4-30test{25*i}.txt' for i in range(11)]
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


# print(get_cleaned_tweets()[:20])
# print("Done!")
#

