import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np



def get_data(file):
    content = []
    tag = []
    for line in file:
        content.append(line[1]+line[2])
        tag.append(line[3].split(' '))
    return content, tag

def tfidf_calculate(title_train, title_test):
    v1 = TfidfVectorizer(max_features = 15000)
    tfidf_train = v1.fit_transform(title_train)
    v2 = TfidfVectorizer(vocabulary = v1.vocabulary_)
    tfidf_test = v2.fit_transform(title_test)
    return tfidf_train, tfidf_test

def get_tag(tag_select, tag):
    tag_matrix = []
    for tags in tag_select:
        tag_tem = []
        for i in range(len(tag)):
            if tags in tag[i]:
                tag_tem.append(1)
            else:
                tag_tem.append(-1)
        tag_matrix.append(tag_tem)
    return tag_matrix

def train_model(tag_matrix_train):
    classifier = []
    for tags in tag_matrix_train:
        clf = SGDClassifier(loss="modified_huber", penalty="l2")
        clf.fit(tfidf_train, tags)
        classifier.append(clf)
    return classifier

def test_model(classifier, tfidf_test):
    predict_test = []
    for i in range(len(classifier)):
        clf = classifier[i]
        predicted = clf.decision_function(tfidf_test)
        predict_test.append(predicted)
    return predict_test


def tag_choice(predict_test, tag_select, outtagfile, outscorefile):
    predict_test = np.array(predict_test)
    tag_select = np.array(tag_select)
    [m,n] = predict_test.shape
    print(predict_test.shape)
    outFile = open(outtagfile, 'w')
    outFile2 = open(outscorefile, 'w')
    for i in range(n):
        slice = predict_test[:,i]
        select = slice[slice> 0]
        if len(select) > 5:
            prediction = tag_select[slice.argsort()[-5:][::-1]]
            score = slice[slice.argsort()[-5:][::-1]]
        elif len(select) == 0:
            prediction = tag_select[slice.argsort()[-1:][::-1]]
            score = slice[slice.argsort()[-1:][::-1]]
        else:
            prediction = tag_select[slice > 0]
            score = slice[slice > 0]
        for word in prediction:
            outFile.write(word + ' ')
        outFile.write('\n')
        for sco in score:
            outFile2.write(str(sco) + ' ')
        outFile2.write('\n')
    outFile.close()
    outFile2.close()


##  load data
train_file = preprocess.line_stream('small_train.csv')
test_file = preprocess.line_stream('small_test.csv')

##  extract title, body and tag
title_train1, tag_train1 = get_data(train_file)
title_test, tag_test = get_data(test_file)

##  sample some train data
title_train = title_train1[0:500000]
tag_train = tag_train1[0:500000]
del title_train1
del tag_train1

##  calculate the total tags
tag_select = preprocess.tag_selection(tag_train)
print(len(tag_select))

## calculate the tfidf
tfidf_train, tfidf_test = tfidf_calculate(title_train, title_test)

print(1)

## calculate the tag matrix
tag_matrix_train = get_tag(tag_select, tag_train)

print(2)


# split the tag models

tag_matrix_train_1 = tag_matrix_train[0:2600]
tag_matrix_train_2 = tag_matrix_train[2600:5200]
tag_matrix_train_3 = tag_matrix_train[5200:7800]
tag_matrix_train_4 = tag_matrix_train[7800:10400]
tag_matrix_train_5 = tag_matrix_train[10400:13000]
del tag_matrix_train
tag_select_1 = tag_select[0:2600]
tag_select_2 = tag_select[2600:5200]
tag_select_3 = tag_select[5200:7800]
tag_select_4 = tag_select[7800:10400]
tag_select_5 = tag_select[10400:13000]
del tag_select


## train model, predict for each tag model
classifier = train_model(tag_matrix_train_1)
print(3)
predict_test = test_model(classifier, tfidf_test)
print(4)
tag_choice(predict_test, tag_select_1, 'prediction_1.txt', 'score_1.txt')
del tag_matrix_train_1
del tag_select_1

del classifier
del predict_test
classifier = train_model(tag_matrix_train_2)
print(5)
predict_test = test_model(classifier, tfidf_test)
print(6)
tag_choice(predict_test, tag_select_2, 'prediction_2.txt', 'score_2.txt')
del tag_matrix_train_2
del tag_select_2



del classifier
del predict_test
classifier = train_model(tag_matrix_train_3)
print(7)
predict_test = test_model(classifier, tfidf_test)
print(8)
tag_choice(predict_test, tag_select_3, 'prediction_3.txt', 'score_3.txt')
del tag_matrix_train_3
del tag_select_3

del classifier
del predict_test
classifier = train_model(tag_matrix_train_4)
print(9)
predict_test = test_model(classifier, tfidf_test)
print(10)
tag_choice(predict_test, tag_select_4, 'prediction_4.txt', 'score_4.txt')
del tag_matrix_train_4
del tag_select_4

del classifier
del predict_test
classifier = train_model(tag_matrix_train_5)
print(11)
predict_test = test_model(classifier, tfidf_test)
print(12)
tag_choice(predict_test, tag_select_5, 'prediction_5.txt', 'score_5.txt')
del tag_matrix_train_5
del tag_select_5