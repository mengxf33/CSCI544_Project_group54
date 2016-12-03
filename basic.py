from collections import defaultdict
from collections import Counter
from preprocess import *

total, common, usefulness = defaultdict(int), defaultdict(int), defaultdict(int)
filename ='../processed_data/processed_train.csv'
# filename ='../tiny_train.csv'
for line in line_stream(filename):
    ID, title, body, tags = line
    title = title_tokenize(title)
    tags = set(tags.split())
    for word in title:
        total[word] += 1
        if word in tags:
            common[word] += 1
for word in total:
    usefulness[word] = common[word] / total[word]

def basic_recommender(title, usefulness=usefulness):
    return Counter({word: usefulness.get(word, 0)
                    for word in title_tokenize(title)})

# title = "R Error Invalid type (list) for variable"
# print(basic_recommender(title))