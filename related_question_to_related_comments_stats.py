import basic_stats
import csv
import spacy

import gensim

import word2vec_utils

ORIGINAL_QUESTION_ID = "question_id"
RELATED_QUESTION_ID = "related_question_id"
RELATED_COMMENT_ID = "related_comment_id"
COMMENT_RELEVANCE = "comment_relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
RELEVANCE = "relevance"
W2V_COSINE_SIMILARITY = "w2v_cosine_similarity"

model = spacy.load('en')
soup = basic_stats.load('data/SemEval2016-Task3-CQA-QL-dev.xml')
original_questions = soup.findAll("OrgQuestion")
word2vec_model = gensim.models.Word2Vec.load('word2vec_model/SemEval2016-Task3-CQA-QL-dev_model')

with open('csv/RelQuestion_to_RelComment_stats.csv', 'w') as csvfile:
    fieldnames = [ORIGINAL_QUESTION_ID, RELATED_QUESTION_ID, RELATED_COMMENT_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE,
                  COSINE_SIMILARITY,W2V_COSINE_SIMILARITY, RELEVANCE]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for original_question in original_questions:
        thread = original_question.find("Thread")
        related_question = thread.find("RelQuestion")
        related_question_body = basic_stats.remove_subject_from_question(related_question.RelQBody.text)
        if related_question_body == "":
            print("invalid data in", related_question["RELQ_ID"], "reason: empty body")
            continue
        related_question_vector = word2vec_utils.sentence_vectors_mean(
            word2vec_utils.sentence2vectors(related_question_body, word2vec_model, lower_case=True))

        related_comments = thread.findAll("RelComment")
        for related_comment in related_comments:
            row = {}
            related_comment_body = related_comment.RelCText.text
            related_comment_vector = word2vec_utils.sentence_vectors_mean(
                word2vec_utils.sentence2vectors(related_comment_body, word2vec_model, lower_case=True))

            row[ORIGINAL_QUESTION_ID] = original_question['ORGQ_ID']
            row[RELATED_QUESTION_ID] = related_question['RELQ_ID']
            row[RELATED_COMMENT_ID] = related_comment['RELC_ID']
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(related_question_body, related_comment_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(related_question_body, related_comment_body)
            row[COSINE_SIMILARITY] = round(
                basic_stats.cosine_similarity(model, related_question_body, related_comment_body), 3)
            row[W2V_COSINE_SIMILARITY] = word2vec_utils.vectors_cosine_similarity(related_question_vector,
                                                                                  related_comment_vector)
            row[RELEVANCE] = related_comment['RELC_RELEVANCE2RELQ']
            writer.writerow(row)
