#!/usr/bin/env python
# coding: utf-8

# ORF 350 HW 7 Georgy Noarov (gnoarov) Collaborators: only myself

# In[300]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import string, re

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import spacy
spacy.load('en_core_web_sm')
lemmatizer = spacy.lang.en.English()

# In[382]:

a = np.load("embeddings.npz",allow_pickle=True)
real_word2ind = {}
foo = a["word2ind"]
word2ind = a["word2ind"].flatten()[0]
for k in word2ind:
    real_word2ind[word2ind[k]] = k
emb = a["emb"]

def get_embedding(word):
    if word in real_word2ind:
        return emb[real_word2ind[word],]
    return np.zeros((128))

#print(get_embedding('sit'))
#print(get_embedding('book'))

def word2vec_avg(doc):
    doc = my_preprocessor(doc)
    tokenized_words = word_tokenize(doc)
    avg_emb = np.zeros((128))
    for word in tokenized_words:
        avg_emb += get_embedding(word)
    return avg_emb/len(tokenized_words)

def word2vec_np(df):
    return np.array([word2vec_avg(doc) for doc in df['text']])

def simple_classifiers_word2vec_pca(df, n_comps):
    df = df.reset_index()
    a, b, c, d = train_test_split(range(1, 41), range(1, 41), test_size=0.2, random_state=42)
    
    reduced_data = PCA(n_comps).fit_transform(word2vec_np(df))
    df_train = df[df['num'].isin(a)]
    df_test = df[df['num'].isin(b)]
    
    X_train = reduced_data[df.index[df['num'].isin(a)]]
    X_test = reduced_data[df.index[df['num'].isin(b)]]
    y_train = df_train['type']
    y_test = df_test['type']
    
#     clf_mnb = MultinomialNB().fit(X_train, y_train)
#     predicted_mnb = clf_mnb.predict(X_test)
#     print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted_mnb))
    
    clf_lreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    predicted_lreg = clf_lreg.predict(X_test)
    print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_lreg))
    
    clf_svm = LinearSVC(random_state=0, tol=1e-5).fit(X_train, y_train)
    predicted_svm = clf_svm.predict(X_test)
    print("Linear SVM Accuracy:", metrics.accuracy_score(y_test, predicted_svm))

# simple_classifiers_word2vec_pca(df_edu, 7)

# labels_color_map = {
#    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
#    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
# }

# fig, ax = plt.subplots()
# colors = iter(cm.rainbow(np.linspace(0, 1, 40)))
# for index, instance in enumerate(reduced_data):
#     # print instance, index, labels[index]
#     pca_comp_1, pca_comp_2 = reduced_data[index]
#     ax.scatter(pca_comp_1, pca_comp_2, color=labels_color_map[df_edu['type'].values[index]])
# plt.show()

# simple_classifiers(df_edu, )


# In[314]:


def print_topfeatures(vectorizer, clf):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    print("Number of features: ", len(feature_names))
    topfeatures_real = np.argsort(clf.coef_[0])[-10:]
    topfeatures_fake = np.argsort(clf.coef_[0])[0:10]
    
    print("Words most indicative of real news: %s" % (" ".join(feature_names[j] for j in topfeatures_real)))
    print("Words most indicative of fake news: %s" % (" ".join(feature_names[j] for j in topfeatures_fake)))

def simple_classifiers(df, vectorizer):
    vectorizer = vectorizer.fit(df['text'])
    
    a, b, c, d = train_test_split(range(1, 41), range(1, 41), test_size=0.2, random_state=42)
    
    df_train = df[df['num'].isin(a)]
    df_test = df[df['num'].isin(b)]
    
    X_train = vectorizer.transform(df_train['text'])
    X_test = vectorizer.transform(df_test['text'])
    y_train = df_train['type']
    y_test = df_test['type']
    
    clf_mnb = MultinomialNB().fit(X_train, y_train)
    predicted_mnb = clf_mnb.predict(X_test)
    print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted_mnb))
    
    clf_lreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    predicted_lreg = clf_lreg.predict(X_test)
    print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_lreg))
    
    clf_svm = LinearSVC(random_state=0, tol=1e-5).fit(X_train, y_train)
    predicted_svm = clf_svm.predict(X_test)
    print("Linear SVM Accuracy:", metrics.accuracy_score(y_test, predicted_svm))
    
    print_top10(vectorizer, clf_svm)
    
    #for i in vectorizer.get_feature_names():
    #    print(i)
    #    print(wm2df(tf_edu, vectorizer.get_feature_names()))
    
def simple_classifiers_for_each_area(vectorizer):
    print("On Education:")
    simple_classifiers(df_edu, vectorizer)
    print()
    print("On Business:")
    simple_classifiers(df_biz, vectorizer)
    print()
    print("On Entertainment:")
    simple_classifiers(df_ent, vectorizer)
    print()
    print("On Sports:")
    simple_classifiers(df_spo, vectorizer)
    print()
    print("On Politics:")
    simple_classifiers(df_pol, vectorizer)
    print()
    print("On Tech:")
    simple_classifiers(df_tec, vectorizer)
    
    return vectorizer.transform(df_edu['text'])


# In[315]:


# this code, used to read in the data, comes from my own work for a different course

# fake news dataset with different genres
directory_real = os.fsencode('./fakeNewsDatasets/fakeNewsDataset/legit')
directory_fake = os.fsencode('./fakeNewsDatasets/fakeNewsDataset/fake')
real_news = os.fsdecode(directory_real)
fake_news = os.fsdecode(directory_fake)

files_real = [[open(real_news + '/' + os.fsdecode(file), 'r', encoding='utf-8-sig').read(),                os.fsdecode(file)[:3], int(''.join(c for c in os.fsdecode(file) if c.isdigit()))]               for file in os.listdir(directory_real)]
files_fake = [[open(fake_news + '/' + os.fsdecode(file), 'r', encoding='utf-8-sig').read(),                os.fsdecode(file)[:3], int(''.join(c for c in os.fsdecode(file) if c.isdigit()))]               for file in os.listdir(directory_fake)]

df_misc = pd.concat([pd.DataFrame([[0, file[0], file[1], file[2]]], columns=['type','text','area', 'num'])                          for file in files_real] + 
                         [pd.DataFrame([[1, file[0], file[1], file[2]]], columns=['type', 'text', 'area', 'num']) \
                          for file in files_fake], ignore_index=True)

# fake news dataset on celebrities
directory_real = os.fsencode('./fakeNewsDatasets/celebrityDataset/legit')
directory_fake = os.fsencode('./fakeNewsDatasets/celebrityDataset/fake')
real_news = os.fsdecode(directory_real)
fake_news = os.fsdecode(directory_fake)

files_real = [open(real_news + '/' + os.fsdecode(file), 'r', encoding='utf-8-sig').read()               for file in os.listdir(directory_real)]
files_fake = [open(fake_news + '/' + os.fsdecode(file), 'r', encoding='utf-8-sig').read()               for file in os.listdir(directory_fake)]

df_celebr = pd.concat([pd.DataFrame([[0, file]], columns=['type', 'text']) for file in files_real] + 
                         [pd.DataFrame([[1, file]], columns=['type', 'text']) for file in files_fake], \
                         ignore_index=True)


# In[316]:


X_train, X_test, y_train, y_test =     train_test_split(range(1, 41), range(1, 41), test_size=0.3, random_state=42)
print(X_train)
print(df_misc['text'].loc[(df_misc['area'] == 'ent') & (df_misc['num'] == 35)])
# let us look at news that are on education and are real
print(df_misc['text'].loc[(df_misc['area'] == 'edu') & (df_misc['type'] == 0)])


# In[317]:


# let us look at the celebrity-themed dataset
print(df_celebr)


# In[318]:


# create dataframe for each subject
df_edu = df_misc.loc[df_misc['area'] == 'edu']
df_biz = df_misc.loc[df_misc['area'] == 'biz']
df_ent = df_misc.loc[df_misc['area'] == 'ent']
df_spo = df_misc.loc[df_misc['area'] == 'spo']
df_pol = df_misc.loc[df_misc['area'] == 'pol']
df_tec = df_misc.loc[df_misc['area'] == 'tec']
#for i in [(str(a) + ' ' + str(b)) for a, b in zip(range(80), df_edu['text'].values)]:
    #print(i)


# In[319]:


# do multinomial naive bayes on tfidf
tf=TfidfVectorizer()
simple_classifiers_for_each_area(tf)


# In[320]:


# do multinomial naive bayes on tf
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)

simple_classifiers_for_each_area(cv)


# In[321]:


import nltk

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


# In[342]:


# create a dataframe from a word matrix: this function found online
def wm2df(wm, feat_names):
    
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)

#from nltk.tokenize import word_tokenize
#token_edu = word_tokenize("jjswfgij asijfpioa poajsfiop")
#print(token_edu)

def my_preprocessor(doc):
    #print(doc)
    #print(' '.join(doc.lower().split()))
    #return ' '.join(doc.lower().split())
    #return re.sub('\n+', ' ', doc.lower())
    s = ' '.join(doc.lower().split())
    #print(re.sub(' +', ' ', re.sub('[0-9]+', '', s.translate(str.maketrans('', '', string.punctuation)))))
    return re.sub(' +', ' ', re.sub('[0-9]+', '', s.translate(str.maketrans('', '', string.punctuation))))

def get_lemma(token):
    if token.lemma_.isdigit():
        return token.text.lower()
    if token.pos_ == 'PRON' or token.lemma_ == '-PRON-':
        return token.text.lower()
    return token.lemma_.lower()

def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    #return [token.lemma_ for token in tokens]
    return [get_lemma(token) for token in tokens]

tfidf = TfidfVectorizer(lowercase=True, strip_accents='unicode',                         tokenizer=my_tokenizer, ngram_range=(1, 1), preprocessor=my_preprocessor)

tfidf_matrix = simple_classifiers_for_each_area(tfidf)
X = tfidf_matrix.todense()

clustering_model = KMeans(
    n_clusters=2,
    precompute_distances="auto",
    n_jobs=-1
)

predicted_kmeans = clustering_model.fit_predict(X)
print(len(predicted_kmeans))
print(metrics.accuracy_score(df_edu['type'], predicted_kmeans))

reduced_data = PCA(n_components=2).fit_transform(X)
# print reduced_data

#labels_color_map = {
#   0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
#   5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
#}

import matplotlib.cm as cm

fig, ax = plt.subplots()
colors = iter(cm.rainbow(np.linspace(0, 1, 40)))
for index, instance in enumerate(reduced_data):
    # print instance, index, labels[index]
    pca_comp_1, pca_comp_2 = reduced_data[index]
    print(index)
    x = df_edu['num'].values
    col = 1/x[index]#labels_color_map[df_edu['type'].values[index]]
    print(col)
    
    #colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))
    #for y in ys:
    #    plt.scatter(x, y, color=next(colors))
    ax.scatter(pca_comp_1, pca_comp_2, color=col)
plt.show()


# In[357]:





# In[ ]:





# In[ ]:




