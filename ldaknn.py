from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize # tokenlize into sentence
import csv
from itertools import chain
import pickle
import numpy as np
from scipy.sparse import vstack
import operator
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
from collections import defaultdict
#from sklearn.feature_extraction.txt import HashingVectorizer
import time

def dd():
    return defaultdict(int)

def f1_score(y_true, y_pred):
    #print(y_true.shape, y_pred.shape)
    tp = np.dot(y_true, y_pred.T)[0,0]
    p = tp / np.sum(y_pred)
    r = tp / np.sum(y_true)
    if abs(tp) < 1e-4:
        return 0, np.sum(y_true)
    return 2 * p * r / (p + r), np.sum(y_true)

def line_stream(filename):
    with open(filename, encoding = "utf8") as csvfile:
        next(csvfile) # skip header
        for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
            yield line

# remove stopwords and tokenize string
#stopwords = set(stopwords.words('english')).union(set(string.punctuation))
#def title_tokenize(title):
#    return [i for i in word_tokenize(title.lower()) if i not in stopwords]

stopwords = set(stopwords.words('english')).union(set(string.punctuation))
def text_stream(filename):
    for line in line_stream(filename):
        yield ' '.join(tokenize(line[1]) )

def tokenize(title):
    return [i for i in word_tokenize(title.lower()) if i not in stopwords]

def tag_stream(filename):
    for line in line_stream(filename):
        yield line[3]


def precision(prediction, truth_stream):
    n = len(prediction)
    tag_counter = CountVectorizer()
    tag_count = tag_counter.fit_transform(chain(iter(prediction) , truth_stream))
    res = 0
    weight = 0
    for y_true, y_pred in zip(tag_count[n:].T, tag_count[:n].T):
        s, w = f1_score(y_true, y_pred)
        res += s*w
        weight += w
    return res/weight


def prediction_with_threshold(distance, indices, threshold):
    prediction = []
    j = -1
    x = 0
    for i, d in zip(indices, distance):
        j+=1
        if j in repeated:
            x+=1
            prediction.append(repeated[j])
            continue
        
        p = []
        for a,b in zip(i,d):
            if b > threshold:
                break
            p.append(tag_list[a][0])
        prediction.append(' '.join(p))
    return prediction
    

train_file = "small_train.csv"
test_file = "small_test.csv"
top_threshold = 8000
n_top_words =10
n_topics = 500

# tf_vectorizer = CountVectorizer()
# tic = time.time()
# tf = tf_vectorizer.fit_transform(text_stream(train_file))
# feature_names = tf_vectorizer.get_feature_names()
# print("tf count time = " , time.time() - tic)



# i=0
# for line in line_stream(train_file):
#     i += 1
#     if i % 10000 == 0:
#         print(i)
#     n = len(line[3].split())
#     title_len[n].append(len(tokenize(line[1])))
#     body_len[n].append(len(tokenize(line[2])))

# with open("length.tmp",'wb') as f:
#     pickle.dump(title_len, f)
#     pickle.dump(body_len, f)

# with open("length.tmp",'rb') as f:
#     title_len = pickle.load(f)
#     body_len = pickle.load(f)

# print(len(title_len))
# for i in range(10):
#     print(i, np.mean(title_len[i]), np.mean(body_len[i]), len(title_len[i]))

# title_len_tag = defaultdict(list)
# body_len_tag = defaultdict(list)
# for i, l in title_len.items():
#     for j in l:
#         title_len_tag[j].append(i)
# for i, l in body_len.items():
#     for j in l:
#         body_len_tag[j].append(i)
# body_len_tag_median = {i:np.median(body_len_tag[i]) for i in body_len_tag}

# title_tag = defaultdict(set)
# for line in line_stream(train_file):
#     title_tag[line[1]].update(line[3].split())

# tic = time.time()
# tag_text = defaultdict(dd)
# tag_count = defaultdict(int)
# i = 0
# print(train_file)
# for title, tag_list in title_tag.items():
#     i+=1
#     if i%10000==0:
#         print(i)
#     for tag in tag_list:
#         for word in tokenize(title):
#             tag_text[tag][word] += 1
#         tag_count[tag] += 1
# print("count time =", time.time()- tic)

# with open("tag_count.tmp","wb") as f:
#     pickle.dump(tag_text,f)
#     pickle.dump(tag_count, f)

# tic = time.time()
# tag_text = defaultdict(dd)
# tag_count = defaultdict(int)
# i = 0
# print(train_file)
# for line in line_stream(train_file):
#     i+=1
#     if i%10000==0:
#         print(i)
#     for tag in line[3].split():
#         for word in tokenize(line[1]):
#             tag_text[tag][word] += 1
#         tag_count[tag] += 1
# print("count time =", time.time()- tic)

# with open("tag_count.tmp","wb") as f:
#     pickle.dump(tag_text,f)
#     pickle.dump(tag_count, f)


with open("tag_count.tmp",'rb') as f:
    tag_text = pickle.load(f)
    tag_count = pickle.load(f)


tag_list = sorted(tag_count.items(), key=operator.itemgetter(1))
tag_list.reverse()
print(tag_list[::1000])
tag_list = tag_list[:top_threshold]


vocabulary = set()
for tag in tag_list:
    word_dict = tag_text[tag[0]]
    word_list = sorted(word_dict.items(), key = operator.itemgetter(1))
    vocabulary.update([word[0] for word in word_list[-max(n_top_words,len(word_list)//20):]])

print(len(vocabulary))
restricted = []
for tag in tag_list:
    d = tag_text[tag[0]]
    restricted.append({k:d[k] for k in d if k in vocabulary})


dict_vect = DictVectorizer()
tag_count_matrix = dict_vect.fit_transform(restricted)

print(tag_count_matrix.shape)

count_vect = CountVectorizer(vocabulary = dict_vect.get_feature_names())

test_count_matrix = count_vect.transform(text_stream(test_file))
print(test_count_matrix.shape)
feature_matrix = vstack((tag_count_matrix, test_count_matrix))

tic = time.time()
transformer = TfidfTransformer()
feature_matrix = transformer.fit_transform(feature_matrix)
print('TFIDF time = ', time.time()-tic)

# tic = time.time()
# nbrs = NearestNeighbors(n_neighbors=5)
# nbrs.fit(feature_matrix[:top_threshold])
# print("neighboor tree time =", time.time()-tic)


# tic = time.time()

# distance, indices = nbrs.kneighbors(feature_matrix[len(tag_list):])
# print('predict time =', time.time()-tic)

# np.save("distance", distance)
# np.save("indices", indices)
distance = np.load("distance.npy")
indices = np.load("indices.npy")


tic = time.time()    

title_tag =  {}
for line in line_stream(train_file):
    title_tag[line[1]] = line[3]
    
repeated = {}
for i, title in enumerate(text_stream(test_file)):
    if title in title_tag:
        repeated[i] = title_tag[title]
with open("repeated.tmp",'wb') as f:
    pickle.dump(repeated, f)
with open("repeated.tmp",'rb') as f:
    repeated = pickle.load(f)
print(len(repeated))


print("repeated find time =", time.time()- tic)

for th in np.arange(1.2, 1.4, 0.02):
    prediction = prediction_with_threshold(distance,indices, th)
    print(th, precision(prediction, tag_stream(test_file)))

#print(precision(open("pred.tmp").readlines(), tag_stream(test_file)))
