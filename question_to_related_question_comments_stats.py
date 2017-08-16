import basic_stats
import csv
import spacy

# CSV file headers
ORIGINAL_QUESTION_ID = "question_id"
RELATED_COMMENT_ID = "related_comment_id"
RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"


model = spacy.load('en')
soup = basic_stats.load('Q1_sample.xml')
original_questions = soup.findAll("OrgQuestion")

with open('OrgQuestion_to_RelComment_stats.csv', 'w') as csvfile:
    fieldnames = [ORIGINAL_QUESTION_ID, RELATED_COMMENT_ID, JACCARD_DISTANCE, LENGTH_DIFFERENCE, COSINE_SIMILARITY, RELEVANCE]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    for original_question in original_questions:
        related_comments = original_question.findAll("RelComment")
        for related_comment in related_comments:
            row = {}
            related_comment_body = related_comment.RelCClean.text
            original_question_body = basic_stats.remove_subject_from_question(original_question.OrgQBody.text)

            row[ORIGINAL_QUESTION_ID] = original_question['ORGQ_ID']
            row[RELATED_COMMENT_ID] = related_comment['RELC_ID']
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(original_question_body, related_comment_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(original_question_body, related_comment_body)
            row[COSINE_SIMILARITY] = round(basic_stats.cosine_similarity(model, original_question_body, related_comment_body), 3)
            row[RELEVANCE] = related_comment['RELC_RELEVANCE2ORGQ']
            writer.writerow(row)
