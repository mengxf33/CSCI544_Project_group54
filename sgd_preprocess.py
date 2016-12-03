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


def tag_selection(tag_train):
    tag_dic = dict()
    outFile = open('tag.txt', 'w')
    for tag in tag_train:
        for t in tag:
            if t in tag_dic:
                tag_dic[t] += 1
            else:
                tag_dic[t] = 1

    tag_select = []
    for key, value in tag_dic.items():
        if value > 5:
            tag_select.append(key)
            outFile.write(key)
            outFile.write('\n')

    outFile.close()
    return tag_select