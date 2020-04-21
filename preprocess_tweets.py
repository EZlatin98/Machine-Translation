# %%

import json
import re


# %%

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

tweets = []
count = 0
bad_format = 0
bad_format_text = []
bad_format_files = ['test0.txt', 'test25.txt', 'test50.txt', 'test75.txt',
         'test100.txt', 'test125.txt', 'test150.txt', 'test175.txt',
         'test200.txt', 'test225.txt', 'test250.txt']
# bad_format_files = []
good_format_files = ['4-21test0.txt']
for file in bad_format_files:
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
            # pass
            line = fp.readline()


for file in good_format_files:
    with open(file, encoding="utf-8") as fp:
        line = fp.readline()
        while line:
            start = line.find("{")
            line = line[start:]
            y = json.loads(line)
            tweets.append(y)
            line = fp.readline()

#         print(line)

#         print(y["full_text"])
#     while line:
#         line = fp.readline()


# %%

print("count", count)
print("bad_format", bad_format)

# %%

# print(bad_format_text[0])
