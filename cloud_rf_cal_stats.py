import ensemble_preprocess
from project_xiangfei import *
import datetime



test_file = ensemble_preprocess.line_stream('small_test.csv')
title_test, tag_test = ensemble_preprocess.tag_selection_from_train(test_file)

f1 = open('prediction_rf_1.txt', 'r')
tag_pred1 = f1.readlines()
f2 = open('prediction_rf_2.txt', 'r')
tag_pred2 = f2.readlines()
f1.close()
f2.close()

tag_pred = []
for i in range(len(tag_pred1)):
    temp = tag_pred1[i] + " " + tag_pred2[i]
    tag_pred.append(temp)

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