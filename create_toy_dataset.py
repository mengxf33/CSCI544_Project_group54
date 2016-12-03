import csv

with open("E:/nlp/all_data/Train.csv", encoding="utf8", newline='') as csvfiler:
    csvreader = csv.reader(csvfiler, delimiter=',', quotechar='"')
    header = next(csvreader)
    with open("E:/nlp/tiny_train.csv",'w',encoding="utf8", newline='') as csvfilew:
        csvwriter = csv.writer(csvfilew, dialect='excel')
        csvwriter.writerow(header)
        for i in range(0,2000000):
            csvwriter.writerow(next(csvreader))

with open("E:/nlp/all_data/Train.csv", encoding="utf8", newline='') as csvfiler:
    csvreader = csv.reader(csvfiler, delimiter=',', quotechar='"')
    header = next(csvreader)
    with open("E:/nlp/tiny_test.csv", 'w', encoding="utf8", newline='') as csvfilew:
        csvwriter = csv.writer(csvfilew, dialect='excel')
        csvwriter.writerow(header)
        for i in range(0, 200000):
            csvwriter.writerow(next(csvreader))

# with open("E:/nlp/all_data/processed_test.csv", encoding="utf8", newline='') as csvfiler:
#     csvreader = csv.reader(csvfiler, delimiter=',', quotechar='"')
#     header = next(csvreader)
#     with open("E:/nlp/small_test.csv", 'w', encoding="utf8", newline='') as csvfilew:
#             csvwriter = csv.writer(csvfilew, dialect='excel')
#             csvwriter.writerow(header)
#             for i in range(0, 100000):
#                 csvwriter.writerow(next(csvreader))


