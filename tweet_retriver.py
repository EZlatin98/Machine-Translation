import random

import tweepy as tw
import json
import csv
# http://docs.tweepy.org/en/latest/api.html#help-methods
# relevant fields - created_at, text - remove ones with truncated = True?
#  'author', 'contributors', 'coordinates', 'created_at',
# 'destroy', 'favorite', 'favorite_count', 'favorited', 'geo', 'id', 'id_str', 'in_reply_to_screen_name', 'in_reply_to_status_id',
# 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'is_quote_status', 'lang', 'metadata', 'parse',
# 'parse_list', 'place', 'retweet', 'retweet_count', 'retweeted', 'retweeted_status', 'retweets', 'source', 'source_url', 'text',
# 'truncated', 'user']

import json

# difference between geo, place,


consumer_key = "PJvQ4PfXTXGhveqtZF9QsNfBK"
consumer_secret = "cp7Du8rZfInMn8eqM0x6QaSqFrmF0HTEAVX9UpCv9kqssuU8ju"
access_token = "1246103360777920515-Vh14BL4b4yGfaR9R0kIU6cPzH0CE97"
access_token_secret = "LzGuOuJr1f3pXq03atX6SSh5SlT0DRHVoe4dojFTXm7lm"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True)

target_states = ['TX', 'NY', 'CA']
query = ' OR '.join(['coronavirus', 'covid-19', 'covid', 'illness', 'mask', 'face-mask', 'wash', 'shelter', 'quarantine', 'ventilator', 'hospital'])

def search_and_print(query, max_items, fp, lat, long, city):
    for tweet in tw.Cursor(api.search, tweet_mode="extended",lang="en", q=query, result_type="recent", include_entities = "false", geocode=f"{lat},{long},25mi").items(max_items):
        fp.write(city + ',' + json.dumps(tweet._json) + '\n')

def __main__():
    random.seed(4)
    fp = open(f'4-30test{0}.txt', 'w', encoding='utf-8')
    for i, line in enumerate(filter_csv()):
        if i < 200:
            continue
        if i > 500:
            break
        if i % 25 == 0:
            fp.close()
            fp = open(f'4-30test{i}.txt', 'w', encoding='utf-8')
        print(i)
        search_and_print(query, 100, fp,line['lat'], line['lng'], f"{line['city']} {line['state_id']}")

def filter_csv():
    new_list = []
    with open('uscities.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for elem in reader:
            if 'state_id' in elem and elem['state_id'] in target_states and float(elem['population']) > 10000:
                new_list.append(elem)

    random.shuffle(new_list)
    return new_list

__main__()