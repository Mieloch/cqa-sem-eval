import gensim
import argparse
import basic_stats
from word2vec_utils import tokenize_to_lower_case

parser = argparse.ArgumentParser()
parser.add_argument('--subtask-e',
                    dest='subtask_e',
                    help='Prepare word embeddings for subtask E (different dataset)',
                    action='store_true')

args = parser.parse_args()

if args.subtask_e:
    soup = basic_stats.load('../data/subtask_e_sample.xml')
else:
    soup = basic_stats.load('../data/Q1_sample.xml')

original_questions = soup.findAll("OrgQuestion")

sentences = []
processed_ids = []
tokens_count = 0
# creating corpus
for original_question in original_questions:
    # original questions
    id = original_question["ORGQ_ID"]
    if id not in processed_ids:
        processed_ids.append(id)
        original_question_body = basic_stats.remove_subject_from_question(
            original_question.OrgQBody.text
        )
        tokenize = tokenize_to_lower_case(original_question_body)
        tokens_count += len(tokenize)
        sentences.append(tokenize)

    # related questions
    related_questions = original_question.findAll("RelQuestion")
    for related_question in related_questions:
        related_question_body = basic_stats.remove_subject_from_question(
            related_question.RelQBody.text)
        tokenize = tokenize_to_lower_case(related_question_body)
        tokens_count += len(tokenize)
        sentences.append(tokenize)

    # related comments
    related_comments = original_question.findAll("RelComment")
    for related_comment in related_comments:
        tokenize = tokenize_to_lower_case(related_comment.RelCText.text)
        # tokenize = tokenize_to_lower_case(related_comment.RelCClean.text)
        tokens_count += len(tokenize)
        sentences.append(tokenize)

    # related answers (only for subtask e)
    if args.subtask_e:
        related_answers = original_question.findAll("RelAnswer")
        for related_answer in related_answers:
            body = related_answer.RelAText.text
            tokenize = tokenize_to_lower_case(body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)


print(tokens_count)
# print(sentences)
model = gensim.models.Word2Vec(sentences, min_count=1, window=3, iter=1000, size=100)
model.save('Q1_model')
