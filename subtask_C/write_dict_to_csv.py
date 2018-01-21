import csv
from subtask_C.data_set import raw_dataset


def WriteDictToCSV(csv_file, dict_data, csv_columns, writeheader=False):
    #try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if writeheader:
            writer.writeheader()
        for data in dict_data:
            writer.writerow(data)
    #except IOError as (errno, strerror):
    #    print("I/O error({0}): {1}".format(errno, strerror))
    return


train_data = raw_dataset("data\SemEval2016-Task3-CQA-QL-train-part1.xml")
test_data = raw_dataset("data\SemEval2016-Task3-CQA-QL-dev.xml")

WriteDictToCSV("subtask_C\\csv_data\\data_not_binarized\\train.csv", train_data, ("orgq", "relc", "relc_orgq_relevance"), writeheader=True)
WriteDictToCSV("subtask_C\\csv_data\\data_not_binarized\\test.csv", test_data, ("orgq", "relc", "relc_orgq_relevance"), writeheader=True)