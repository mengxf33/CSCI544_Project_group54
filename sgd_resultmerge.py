import numpy as np
import heapq

prediction_1 = open('prediction_1.txt','r', encoding = "utf8")
prediction_2 = open('prediction_2.txt','r', encoding = "utf8")
prediction_3 = open('prediction_3.txt','r', encoding = "utf8")
prediction_4 = open('prediction_4.txt','r', encoding = "utf8")
prediction_5 = open('prediction_5.txt','r', encoding = "utf8")

score_1 = open('score_1.txt','r', encoding = "utf8")
score_2 = open('score_2.txt','r', encoding = "utf8")
score_3 = open('score_3.txt','r', encoding = "utf8")
score_4 = open('score_4.txt','r', encoding = "utf8")
score_5 = open('score_5.txt','r', encoding = "utf8")


prediction_1 = prediction_1.readlines()
prediction_2 = prediction_2.readlines()
prediction_3 = prediction_3.readlines()
prediction_4 = prediction_4.readlines()
prediction_5 = prediction_5.readlines()
score_1 = score_1.readlines()
score_2 = score_2.readlines()
score_3 = score_3.readlines()
score_4 = score_4.readlines()
score_5 = score_5.readlines()
prediction = []
score = []



outFile = open('prediction_whole.txt', 'w')
outFile2 = open('score_whole.txt', 'w')
for i in range(len(prediction_1)):
    words = prediction_1[i].strip()+ ' ' + prediction_2[i].strip() + ' ' + prediction_3[i].strip() + ' ' +prediction_4[i].strip() + ' ' +prediction_5[i].strip()
    sco = score_1[i].strip() + ' ' + score_2[i].strip() + ' ' + score_3[i].strip() + ' ' + score_4[i].strip()+ ' ' + score_5[i].strip()
    slice_prediction = words.split(' ')
    sliced = sco.split(' ')
    if len(sliced) > 5:
        select = heapq.nlargest(5, sliced)
        prediction1 = []
        score1 = []
        for score in select:
            id = sliced.index(score)
            prediction1.append(slice_prediction[id])
            score1.append(sliced[id])
    else:
        prediction1 = slice_prediction
        score1 = sliced
    for word in prediction1:
        outFile.write(word + ' ')
    outFile.write('\n')
    for sco in score1:
        outFile2.write(str(sco) + ' ')
    outFile2.write('\n')
outFile.close()
outFile2.close()


