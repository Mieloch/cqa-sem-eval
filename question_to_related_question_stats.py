import basic_stats
import csv
import spacy

# CSV file headers
RELATED_QUESTION_ID = "related_question_id"
ORGINAL_QUESTION_ID = "original_id"
RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"

model = spacy.load('en')
soup = basic_stats.load('data/Q1_sample.xml')
original_questions = soup.findAll("OrgQuestion")

with open('csv/OrgQuestion_to_RelQuestion_stats.csv', 'w') as csvfile:
    fieldnames = [ORGINAL_QUESTION_ID, RELATED_QUESTION_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE, COSINE_SIMILARITY,
                  RELEVANCE]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for original_question in original_questions:
        related_questions = original_question.findAll("RelQuestion")
        for related_question in related_questions:
            row = {}
            related_question_body = basic_stats.remove_subject_from_question(related_question.RelQBody.text)

            orginal_question_body = basic_stats.remove_subject_from_question(original_question.OrgQBody.text)
            row[ORGINAL_QUESTION_ID] = original_question['ORGQ_ID']
            row[RELATED_QUESTION_ID] = related_question['RELQ_ID']
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(orginal_question_body, related_question_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(orginal_question_body, related_question_body)
            row[COSINE_SIMILARITY] = round(
                basic_stats.cosine_similarity(model, orginal_question_body, related_question_body), 3)
            row[RELEVANCE] = related_question['RELQ_RELEVANCE2ORGQ']
            writer.writerow(row)
