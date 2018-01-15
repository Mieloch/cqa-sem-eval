import csv

import numpy as np
from bs4 import BeautifulSoup

xml_files = ["../data/SemEval2016-Task3-CQA-QL-dev.xml", "../data/SemEval2016-Task3-CQA-QL-train-part1.xml",
             "../data/SemEval2016-Task3-CQA-QL-train-part2.xml"]
RELATED_QUESTION_ID = "related_question_id"
RELATED_COMMENT_ID = "related_comment_id"
QUESTION_TEXT = "related_question_text"
COMMENT_TEXT = "related_comment_text"
RELEVANCE = "relevance"

# open validation data set csv
validation_data_set_csv = open('csv/validation_data_set.csv', 'a', encoding="utf-8")
fieldnames = [RELATED_QUESTION_ID, RELATED_COMMENT_ID, QUESTION_TEXT, COMMENT_TEXT, RELEVANCE]
validation_data_writer = csv.DictWriter(validation_data_set_csv, fieldnames=fieldnames, lineterminator='\n')
validation_data_writer.writeheader()

# open train data set csv
test_data_set_csv = open('csv/train_data_set.csv', 'a', encoding="utf-8")
fieldnames = [RELATED_QUESTION_ID, RELATED_COMMENT_ID, QUESTION_TEXT, COMMENT_TEXT, RELEVANCE]
train_data_writer = csv.DictWriter(test_data_set_csv, fieldnames=fieldnames, lineterminator='\n')
train_data_writer.writeheader()

for xml_file_name in xml_files:
    with open(xml_file_name, 'r', encoding="utf8") as xml_file:
        rows = []
        soup = BeautifulSoup(xml_file.read(), "xml")
        threads = soup.findAll('Thread', recursive=True)
        for index, thread in enumerate(threads):
            print("Thread {}/{} in {}".format(index, len(threads), xml_file_name))
            thread_soup = BeautifulSoup(str(thread), "xml")
            related_question_text = thread_soup.RelQuestion.RelQBody.text
            if related_question_text == '':
                print("WARN! empty question_text")
                continue
            comments = thread_soup.findAll("RelComment")
            for comment in comments:
                comment_text = comment.RelCText.text
                if comment_text == '':
                    print("WARN! empty comment")
                    continue
                row = {}
                row[RELATED_QUESTION_ID] = thread_soup.RelQuestion['RELQ_ID']
                row[RELATED_COMMENT_ID] = comment['RELC_ID']
                row[QUESTION_TEXT] = related_question_text
                row[COMMENT_TEXT] = comment_text
                row[RELEVANCE] = comment['RELC_RELEVANCE2RELQ']
                rows.append(row)
        np.random.shuffle(rows)
        percent_of_samples = round(len(rows) * 0.8)
        training, test = rows[:percent_of_samples], rows[percent_of_samples:]
        validation_data_writer.writerows(test)
        train_data_writer.writerows(training)

validation_data_set_csv.close()
test_data_set_csv.close()
