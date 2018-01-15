import csv

import pandas as pd
import spacy

import basic_stats

RELATED_QUESTION_ID = "related_question_id"
RELATED_COMMENT_ID = "related_comment_id"
QUESTION_TEXT = "related_question_text"
COMMENT_TEXT = "related_comment_text"
RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
DATA_SET_FILE = "train_data_set.csv"
model = spacy.load('en')


def label_to_class(label):
    if label == "Good":
        return 1
    elif label == "PotentiallyUseful":
        return 0
    elif label == "Bad":
        return 0


with open('statistic_metrics.csv', 'w') as csvfile:
    fieldnames = [RELATED_QUESTION_ID, RELATED_COMMENT_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE,
                  COSINE_SIMILARITY, RELEVANCE]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    data_set = pd.read_csv(DATA_SET_FILE)
    for index, data_set_row in data_set.iterrows():
        if index % 100 == 0:
            print("processing data_set_row {}/{}".format(index, len(data_set)))
        row = {}
        comment_text = data_set_row[COMMENT_TEXT]
        question_text = data_set_row[QUESTION_TEXT]
        row[RELATED_QUESTION_ID] = data_set_row[RELATED_QUESTION_ID]
        row[RELATED_COMMENT_ID] = data_set_row[RELATED_COMMENT_ID]
        row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(comment_text, question_text), 3)
        row[LENGTH_DIFFERENCE] = basic_stats.length_difference(comment_text, question_text)
        row[COSINE_SIMILARITY] = round(
            basic_stats.cosine_similarity(model, comment_text, question_text), 3)
        row[RELEVANCE] = label_to_class(data_set_row[RELEVANCE])
        writer.writerow(row)
