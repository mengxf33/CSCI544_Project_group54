from __future__ import division
import pandas as pd
import collections, hashlib, math, operator
from preprocess import *

# string hashing
def hash_str(str, nr_bins):
    try:
        return int(hashlib.md5(str.encode('utf-8')).hexdigest(), 16) % (nr_bins - 1) + 1
    except AttributeError:
        return -1

# merge two dicts
def merge_dicts(x, y):
    if len(x) < len(y):
        return merge_dicts(y, x)
    for i in y:
        if i in x:
            x[i] += y[i]
        else:
            x[i] = y[i]
    return x

# model training
train_data_path = "../small_train.csv"
# key: tag; value: dict with word as key and corresponding count as value
tag_word_dict = collections.defaultdict(dict)
# key: tag; value: probability
tag_prob_dict = collections.defaultdict(int)
# key: question; value: tags
ques_tag_dict = dict()
total_num_tags = 0
corpus = set()
for line in line_stream(train_data_path):
    ID, title, body, tags = line
    b = hash_str(body, math.pow(10, 10))
    t = hash_str(title, math.pow(10, 10))
    ques_tag_dict[(b,t)] = tags
    try:
        title = title.split()
    except AttributeError:
        title = []
    try:
        body = body.split()
    except AttributeError:
        body = []
    tags_list = tags.split()
    temp_dict = collections.defaultdict(int)
    for t in title:
        temp_dict[t] += 5
        corpus.add(t)
    for b in body:
        temp_dict[b] += 1
        corpus.add(b)
    for tag in tags_list:
        tag_word_dict[tag] = merge_dicts(tag_word_dict[tag].copy(), temp_dict.copy())
        tag_prob_dict[tag] += 1
        total_num_tags += 1
    print("Train - " + str(ID))

# key: tag; value: word count
total_count = dict()
for tag in tag_word_dict:
    n = 0
    for word in tag_word_dict[tag]:
        n += tag_word_dict[tag][word]
    total_count[tag] = n
    tag_prob_dict[tag] = tag_prob_dict[tag] / total_num_tags

# model testing
tp = 0
fp = 0
fn = 0
k = 3 # number of tags to predict
test_data_path = "../small_test.csv"
test_data = pd.read_csv(test_data_path, sep= ',')
for index, row in test_data.iterrows():
    # Hash
    b = hash_str(row["Body"], math.pow(10, 10))
    t = hash_str(row["Title"], math.pow(10, 10))
    # If train has the same questions, just use train tags
    true_tags = set(row["Tags"].split())
    if (b,t) in ques_tag_dict:
        predicted_tags = ques_tag_dict[(b,t)]
    else:
        try:
            title = row["Title"].split()
        except AttributeError:
            title = []
        try:
            body = row["Body"].split()
        except AttributeError:
            body = []
        prob = []
        for tag in tag_word_dict:
            prob_temp = math.log(tag_prob_dict[tag])
            words = tag_word_dict[tag]
            for t in title:
                if t in words:
                    pp = (words[t] + 1) / (total_count[tag] + len(corpus))
                else:
                    pp = 1 / (total_count[tag] + len(corpus))
                prob_temp += math.log(pp)
            for b in body:
                if b in words:
                    pp = (words[b] + 1) / (total_count[tag] + len(corpus))
                else:
                    pp = 1 / (total_count[tag] + len(corpus))
                prob_temp += math.log(pp)
            prob.append((tag, prob_temp))
        prob = sorted(prob, key=operator.itemgetter(1), reverse=True)[:k]
        predicted_tags =set([i for i, j in prob])
    for tag in predicted_tags:
        if tag not in true_tags:
            fp += 1
        else:
            tp += 1
    for tag in true_tags:
        if tag not in predicted_tags:
            fn += 1
    print ("Test - " + str(index))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print('precision is', precision)
print('recall is', recall)
print('F1 score is', 2 * precision * recall / (precision + recall))