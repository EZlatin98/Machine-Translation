#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import re
import pandas as pd
import numpy as np


# In[2]:


def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
tweets = []
count = 0
bad_format = 0
bad_format_text = []
files = ['test0.txt', 'test25.txt', 'test50.txt','test75.txt',
         'test100.txt','test125.txt', 'test150.txt','test175.txt',
         'test200.txt','test225.txt','test250.txt']
good_format_files = [f'4-21test{25*i}.txt' for i in range(11)]
for file in files:
    with open(file, encoding="utf-8") as fp:
        line = fp.readline()
        while line:
            count += 1
            line = line.split(",{")
            line = "{" + line[1]
            line = deEmojify(line.replace("\'", "\""))
            line = line.replace("False", "\"False\"").replace("True", "\"True\"").replace("None", "\"None\"")
            line = line.replace("href=\"http:", "href='http:").replace("\\xa0", " ")
            line = re.sub('\"source.*?,', '', line)
            try:
                y = json.loads(line)
                tweets.append(y)
            except:
                bad_format += 1
                bad_format_text.append(line)
                pass
            line = fp.readline()
#         print(line)
        
#         print(y["full_text"])
    #     while line:
    #         line = fp.readline()


for file in good_format_files:
    with open(file, encoding="utf-8") as fp:
        line = fp.readline()
        while line:
            start = line.find("{")
            line = line[start:]
            y = json.loads(line)
            tweets.append(y)
            line = fp.readline()


# In[3]:


print("count", count)
print("bad_format", bad_format)


# In[4]:


print(bad_format_text[0])


# In[ ]:





# In[12]:


county_pop = pd.read_csv("county_pop.csv")
county_pop.head()


# In[ ]:


# In[14]:


# https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
confirmed_covid = pd.read_csv("confirmed_covid.csv")
confirmed_covid.head()
confirmed_4_21 = confirmed_covid[['County Name', 'State', '4/21/20']]
confirmed_4_21.head()

