import basic_stats
import csv
import spacy
import gensim

# CSV file headers
import word2vec_utils

RELATED_QUESTION_ID = "related_question_id"
ORGINAL_QUESTION_ID = "original_id"
RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
BIGRAM_SIMILARITY = "bigram_similarity"
W2V_COSINE_SIMILARITY = "w2v_cosine_similarity"

model = spacy.load('en')
soup = basic_stats.load('data/SemEval2016-Task3-CQA-QL-dev.xml')
original_questions = soup.findAll("OrgQuestion")
word2vec_model = gensim.models.Word2Vec.load('word2vec_model/Q1_model')

with open('csv/OrgQuestion_to_RelQuestion_stats.csv', 'w') as csvfile:
    fieldnames = [ORGINAL_QUESTION_ID, RELATED_QUESTION_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE, COSINE_SIMILARITY,
                  BIGRAM_SIMILARITY,
                  RELEVANCE]
    writer = csv.DictWriter(
        csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for original_question in original_questions:
        related_questions = original_question.findAll("RelQuestion")
        for related_question in related_questions:
            row = {}
            related_question_body = basic_stats.remove_subject_from_question(
                related_question.RelQBody.text)
            related_question_vector = word2vec_utils.sentence_vectors_mean(
                word2vec_utils.sentence2vectors(related_question_body, word2vec_model, lower_case=True))

            original_question_body = basic_stats.remove_subject_from_question(
                original_question.OrgQBody.text)
            original_question_vector = word2vec_utils.sentence_vectors_mean(
                word2vec_utils.sentence2vectors(original_question_body, word2vec_model, lower_case=True))

            row[ORGINAL_QUESTION_ID] = original_question['ORGQ_ID']
            row[RELATED_QUESTION_ID] = related_question['RELQ_ID']
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(
                original_question_body, related_question_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(
                original_question_body, related_question_body)
            row[COSINE_SIMILARITY] = round(
                basic_stats.cosine_similarity(model, original_question_body, related_question_body), 3)
            row[BIGRAM_SIMILARITY] = round(basic_stats.ngram_similarity(
                original_question_body, related_question_body, n=2), 3)
            row[W2V_COSINE_SIMILARITY] = word2vec_utils.vectors_cosine_similarity(original_question_vector,
                                                                                  related_question_vector)

            row[RELEVANCE] = related_question['RELQ_RELEVANCE2ORGQ']
            writer.writerow(row)
