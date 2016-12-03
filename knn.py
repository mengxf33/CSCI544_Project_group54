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


train_file = "small_train.csv"
test_file = "small_test.csv"
top_threshold = 5000

tic = time.time()
tag_text = defaultdict(dd)
tag_count = defaultdict(int)
i = 0
print(train_file)
for line in line_stream(train_file):
    i+=1
    if i%10000==0:
        print(i)
    for tag in line[3].split():
        for word in tokenize(line[1]):
            tag_text[tag][word] += 1
        tag_count[tag] += 1
print("count time =", time.time()- tic)

with open("tag_count.tmp","wb") as f:
    pickle.dump(tag_text,f)
    pickle.dump(tag_count, f)

with open("tag_count.tmp",'rb') as f:
    tag_text = pickle.load(f)
    tag_count = pickle.load(f)

tag_list = sorted(tag_count.items(), key=operator.itemgetter(1))
tag_list.reverse()
print(tag_list[::1000])
tag_list = tag_list[:top_threshold]

dict_vect = DictVectorizer()

tag_count_matrix = dict_vect.fit_transform([tag_text[tag[0]] for tag in tag_list])

print(tag_count_matrix.shape)

count_vect = CountVectorizer(vocabulary = dict_vect.get_feature_names())

test_count_matrix = count_vect.transform(text_stream(test_file))
print(test_count_matrix.shape)
feature_matrix = vstack((tag_count_matrix, test_count_matrix))

tic = time.time()
transformer = TfidfTransformer()
feature_matrix = transformer.fit_transform(feature_matrix)
print('TFIDF time = ', time.time()-tic)

tic = time.time()
nbrs = NearestNeighbors(n_neighbors=5)
nbrs.fit(feature_matrix[:top_threshold])
print("neighboor tree time =", time.time()-tic)

tic = time.time()
_, indices = nbrs.kneighbors(feature_matrix[len(tag_list):])
print('predict time =', time.time()-tic)

prediction = [' '.join([tag_list[i][0] for i in p]) for p in indices]
with open("pred.tmp","w") as f:
    for p in prediction:
        f.write(p+"\n")

print(precision(prediction, tag_stream(test_file)))

#print(precision(open("pred.tmp").readlines(), tag_stream(test_file)))
