from nltk import word_tokenize
from nltk.corpus import stopwords
import csv
import string


def line_stream(filename):
    with open(filename, encoding = "utf8") as csvfile:
        next(csvfile) # skip header
        for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
            yield line

# remove stopwords and tokenize string
stopwords = set(stopwords.words('english')).union(set(string.punctuation))
def title_tokenize(title):
    return [i for i in word_tokenize(title.lower()) if i not in stopwords]


def tag_selection_from_train(tag_train):
    tag_dic = dict()

    for tag in tag_train:
        tags = tag.split(' ')
        for t in tags:
            if t in tag_dic:
                tag_dic[t] += 1
            else:
                tag_dic[t] = 1
    tag_select = []
    for key, value in tag_dic.items():
        if value > 10:
            tag_select.append(key)
    return tag_select

    def get_feature_and_tag(file):
    feature = []
    tag = []
    for line in file:
        # unigram of title + bigram of body
        feature.append(line[1] + build_bi_gram(line[2]))
        tag.append(line[3])
    return feature, tag


def build_bi_gram(line):
    temp_list = line.split()
    bi_gram = []
    for i in range(len(temp_list) - 1):
        bi_gram.append(temp_list[i] + " " + temp_list[i + 1])
    return bi_gram