CSCI 544 - Group 54
Identify Keywords and Tags from Millions of Text Questions

This repository contains all the codes we implemented to do this project. Detailed description of each python script is as following.

########################################################################################################################################
create_toy_dataset.py: sample small training and testing sets.

preprocess.py: utilities to read data and preprocess title including tokenization, stop words removal and punctuation removal.

naive_bayes.py: train a Naive Bayes model and output precision, recall and F1 score.

basic.py: train a basic recommender.

title_to_tag.py: train a title to tag recommender.

tag_to_tag.py: train a tag to tag recommender.

combine.py: combine the results of basic recommender, title to tag recommender and tag to tag recommender. Output precision, recall and F1 score of Tag Recommendation model.

knn.py: train the model using knn, make prediction and get evaluation.

ldaknn.py: use lda get the keywords, then train the model using knn, make prediction and get evaluation. this code is also used to turn parameters.

final.py: the model that give the best prediction result.

sgd_preprocess.py: Do the preprocessing for SGD classifier.

sgd.py: Main code to conduct the SGD classifier, we need to change the loss function to conduct different classifiers.

sgd_resultmerge.py: This code merge the prediction result of the tag batches and generate the final prediction result.

sgd_evalutaion.py: Do the evaluation for SGD classifier method.

ensemble_precess.py: Preprocessing for ensemble methods - Adaboost and random forest.

adaboost.py: Conduct Adaboost classification, different base estimator can be changed by commenting out and uncommenting one line of code.

adaboost_cal_stats.py: Calculate the results for Adaboost method.

cloud_random_forest.py: Note this code runs in python 2.7, since Google cloud compute engine only supports this version of python. It conducts random forest classification.

cloud_rf_cal_stats.py: Note this code runs in python 2.7, since Google cloud compute engine only supports this version of python. Calculate the results for random forest method.


If you have any question, please feel free to contact any of us!

Group 54
Xiangfei Meng
Yuan Jin
He Luan
Ye Wang

December 2nd, 2016
