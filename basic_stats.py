from bs4 import BeautifulSoup
import re
import nltk
import csv
import spacy


RELATED_QUESTION_ID = "related_question_id"

ORGINAL_QUESTION_ID = "original_id"

RELEVANCE = "relevance"

LENGTH_DIFFERENCE = "length_difference"

JACCARD_DISTANCE = "jaccard_distance"

COSINE_SIMILARITY = "cosine_similarity"


def load(file_name):
    with open(file_name, 'r') as myfile:
        return BeautifulSoup(myfile.read(), "xml")


def remove_subject_from_question(question):
    return re.sub('.+\/\/\ ', '', question)


def length_difference(original, related):
    return abs(len(original) - len(related))


def jaccard_distance(original, related):
    org_tokens = set(nltk.word_tokenize(original))
    rel_tokens = set(nltk.word_tokenize(related))

    return nltk.jaccard_distance(org_tokens, rel_tokens)

def cosine_similarity(model, original, related):
    org_tokens = model(original)
    rel_tokens = model(related)
    return org_tokens.similarity(rel_tokens)

model = spacy.load('en')

soup = load('SemEval2016-Task3-CQA-QL-train-part1.xml')
original_questions = soup.findAll("OrgQuestion")

with open('train_set_stat_part1.csv', 'w') as csvfile:
    fieldnames = [ORGINAL_QUESTION_ID, RELATED_QUESTION_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE, COSINE_SIMILARITY, RELEVANCE]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for original_question in original_questions:
        related_questions = original_question.findAll("RelQuestion")
        for related_question in related_questions:
            row = {}
            related_question_body = remove_subject_from_question(related_question.RelQBody.text)

            orginal_question_body = remove_subject_from_question(original_question.OrgQBody.text)
            row[ORGINAL_QUESTION_ID] = original_question['ORGQ_ID']
            row[RELATED_QUESTION_ID] = related_question['RELQ_ID']
            row[JACCARD_DISTANCE] = round(jaccard_distance(orginal_question_body, related_question_body), 3)
            row[LENGTH_DIFFERENCE] = length_difference(orginal_question_body, related_question_body)
            row[COSINE_SIMILARITY] = round(cosine_similarity(model, orginal_question_body, related_question_body), 3)
            row[RELEVANCE] = related_question['RELQ_RELEVANCE2ORGQ']
            writer.writerow(row)
