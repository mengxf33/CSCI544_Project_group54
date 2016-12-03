import preprocess

def get_data(file):
    tag = []
    for line in file:
        tag.append(line[3])
    return tag

def result(predictionfile, truefile):
    file = open(predictionfile,'r')
    lines = file.readlines()
    predict = []
    for line in lines:
        line = line.strip()
        words = line.split(' ')
        predict.append(words)

    test_file = preprocess.line_stream(truefile)

    true = get_data(test_file)
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(true)):
        for tag in predict[i]:
            if tag not in true[i]:
                fp += 1
            else:
                tp += 1
        for tag in true[i]:
            if tag not in predict[i]:
                fn += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    print('precision is', p)
    print('recall is', r)
    print('F1 is', 2 * p * r / (p + r))


result('prediction_whole.txt', 'small_test.csv')
