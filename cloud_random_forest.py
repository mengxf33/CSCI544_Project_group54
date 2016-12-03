import ensemble_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def tfidf_calculate(title_train, title_test):
    v1 = TfidfVectorizer(max_features = 10000)
    tfidf_train = v1.fit_transform(title_train)
    v2 = TfidfVectorizer(vocabulary = v1.vocabulary_)
    tfidf_test = v2.fit_transform(title_test)
    return tfidf_train, tfidf_test


def get_tag(tag_select, tag):
    tag_matrix = []
    words_list = []
    for temp in tag:
        words_list.append(temp.split())
    for tags in tag_select:
        tag_tem = []
        for i in range(len(tag)):
            if tags in words_list[i]:
                tag_tem.append(1)
            else:
                tag_tem.append(-1)
        tag_matrix.append(tag_tem)
    return tag_matrix


def train_model(tag_matrix_train):
    classifier = []
    for tags in tag_matrix_train:
        # since we have 16 CPUs on VM, we set n_jobs=16
        clf = RandomForestClassifier(n_estimators=50, n_jobs=16)
        clf.fit(tfidf_train, tags)
        classifier.append(clf)
    return classifier


def test_model(classifier, tfidf_test):
    predict_test = []
    for i in range(len(classifier)):
        clf = classifier[i]
        predicted = clf.predict_proba(tfidf_test)
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
        temp_prediction = tag_select[slice.argsort()[-3:][::-1]]
        temp_score = slice[slice.argsort()[-3:][::-1]]
        prediction = [temp_prediction[0]]
        score = [temp_score[0]]
        best = temp_score[0]
        for j in range(1, len(temp_prediction)):
            if best - temp_score[j] > 0.2:
                break
            prediction.append(temp_prediction[j])
            score.append(temp_score[j])

        for word in prediction:
            outFile.write(word + ' ')
        outFile.write('\n')
        for sco in score:
            outFile2.write(str(sco) + ' ')
        outFile2.write('\n')
    outFile.close()
    outFile2.close()


# load data
train_file = ensemble_preprocess.line_stream('small_train.csv')
test_file = ensemble_preprocess.line_stream('small_test.csv')

# extract title, body and tag
title_train, tag_train = ensemble_preprocess.get_feature_and_tag(train_file)
title_test, tag_test = ensemble_preprocess.get_feature_and_tag(test_file)

del train_file
del test_file

# calculate the total tags
tag_select = ensemble_preprocess.tag_selection(tag_train)

# calculate the tfidf
tfidf_train, tfidf_test = tfidf_calculate(title_train, title_test)

print("tf-idf vecotrs done...")

## calculate the tag matrix
tag_matrix_train = get_tag(tag_select, tag_train)
#tag_matrix_test = get_tag(tag_select, tag_test)

print("Tag matrix done...")


# split the tag models

tag_matrix_train_1 = tag_matrix_train[0:int(len(tag_matrix_train)/2)]
tag_matrix_train_2 = tag_matrix_train[int(len(tag_matrix_train)/2):len(tag_matrix_train)]

tag_select_1 = tag_select[0:int(len(tag_matrix_train)/2)]
tag_select_2 = tag_select[int(len(tag_matrix_train)/2):len(tag_matrix_train)]
del tag_matrix_train
del tag_select

## train model, predict for each tag model
classifier = train_model(tag_matrix_train_1)
print("first batch tags training done...")
predict_test = test_model(classifier, tfidf_test)
print("first batch tags testing done...")
tag_choice(predict_test, tag_select_1, 'prediction_rf_1.txt', 'score_rf_1.txt')
del tag_matrix_train_1
del tag_select_1

del classifier
del predict_test
classifier = train_model(tag_matrix_train_2)
print("second batch tags training done...")
predict_test = test_model(classifier, tfidf_test)
print("second batch tags testing done...")
tag_choice(predict_test, tag_select_2, 'prediction_rf_2.txt', 'score_rf_2.txt')
del tag_matrix_train_2
del tag_select_2
