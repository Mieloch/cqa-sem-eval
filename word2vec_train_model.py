import gensim
import nltk
import basic_stats
from word2vec_utils import tokenize_to_lower_case

soup = basic_stats.load('data/SemEval2016-Task3-CQA-QL-dev.xml')

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
        tokens_count += len(tokenize)
        sentences.append(tokenize)

print(tokens_count)
# print(sentences)
model = gensim.models.Word2Vec(sentences, min_count=1, window=3, iter=100000, size=100)
model.save('word2vec_model/SemEval2016-Task3-CQA-QL-dev_model')
