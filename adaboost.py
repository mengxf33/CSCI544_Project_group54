import ensemble_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import numpy as np


def tfidf_calculate(title_train, title_test):
    v1 = TfidfVectorizer(max_features = 12000)
    tfidf_train = v1.fit_transform(title_train)
    v2 = TfidfVectorizer(vocabulary = v1.vocabulary_)
    tfidf_test = v2.fit_transform(title_test)
    return tfidf_train, tfidf_test


def cal_tag_matrix(tag_select, tag):
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



def train_model(tfidf_train, tag_matrix_train):
    classifier = []
    for tags in tag_matrix_train:
        # SVM as base classifier
        clf = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='rbf'), n_estimators=10)
        # by default, decision tree as base classifier
        # clf = AdaBoostClassifier(n_estimators=20)
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
        temp_prediction = tag_select[slice.argsort()[-5:][::-1]]
        temp_score = slice[slice.argsort()[-5:][::-1]]
        prediction = [temp_prediction[0]]
        score = [temp_score[0]]
        best = temp_score[0]
        for j in range(1, len(temp_prediction)):
            if best - temp_score[j] > 2:
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
tag_select = ensemble_preprocess.tag_selection_from_train(tag_train)

# calculate the tfidf
tfidf_train, tfidf_test = tfidf_calculate(title_train, title_test)

print("tf-idf vecotrs done...")

# calculate the tag matrix
tag_matrix_train = cal_tag_matrix(tag_select, tag_train)

print("Tag matrix done...")


# train model, predict for each tag model
classifier = train_model(tfidf_train, tag_matrix_train)
print("Training done...")
predict_test = test_model(classifier, tfidf_test)
print("Testing done...")
tag_choice(predict_test, tag_select, 'prediction_adaboost.txt', 'score_adaboost.txt')

