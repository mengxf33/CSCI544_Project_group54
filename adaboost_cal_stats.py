import ensemble_preprocess
from project_xiangfei import *
import datetime




test_file = ensemble_preprocess.line_stream('small_test.csv')
title_test, tag_test = ensemble_preprocess.get_feature_and_tag(test_file)

f = open('prediction_adaboost.txt', 'r')
tag_pred = f.readlines()

true_pos = 0
precision_denominator = 0
recall_denominator = 0

for i in range(len(tag_test)):
    line_tags = tag_test[i].split()
    line_pred_tags = tag_pred[i].split()
    precision_denominator += len(line_pred_tags)
    recall_denominator += len(line_tags)
    for temp in line_pred_tags:
        if temp in line_tags:
            true_pos += 1


precision = true_pos / precision_denominator
recall = true_pos / recall_denominator
F1 = 2 * precision * recall / (precision + recall)
print("Precision = " + str(precision))
print("Recall = " + str(recall))
print("F1 = " + str(F1))

print(str(datetime.datetime.today()))