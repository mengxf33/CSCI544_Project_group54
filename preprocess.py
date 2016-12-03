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
