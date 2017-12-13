import gensim
import basic_stats
from word2vec_utils import tokenize_to_lower_case
from lxml import etree
import os

def train_model(data_src, model_name, window=5, iterations=10, workers=4, verbose=False):
    # Sentences here
    sentences = []
    processed_ids = []
    tokens_count = 0
    soup = basic_stats.load(data_src)
    original_questions = soup.findAll("OrgQuestion")

    for original_question in original_questions:
        id = original_question.get("ORGQ_ID")

        if id not in processed_ids:
            processed_ids.append(id)
            original_question_body = original_question.OrgQBody.text
            original_question_body = basic_stats.remove_subject_from_question(
                original_question_body
            )
            tokenize = tokenize_to_lower_case(original_question_body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)

        # related questions
        related_questions = original_question.findAll("RelQuestion")
        for related_question in related_questions:
            related_question_body = related_question.RelQBody.text
            related_question_body = basic_stats.remove_subject_from_question(
                related_question_body)
            tokenize = tokenize_to_lower_case(related_question_body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)

        # related comments
        related_comments = original_question.findAll("RelComment")
        for related_comment in related_comments:
            related_comment_body = related_comment.RelCText.text
            tokenize = tokenize_to_lower_case(related_comment_body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)

        # related answers (only for subtask e)
        related_answers = original_question.findAll("RelAnswer")
        for related_answer in related_answers:
            related_answer_body = related_answer.RelAText.text
            tokenize = tokenize_to_lower_case(related_answer_body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)

    model = gensim.models.Word2Vec(
        sentences, min_count=5, window=window, iter=iterations, size=100, workers=workers)
    model.save(model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        dest='data',
                        help='Training data path (default = data/Q1_sample.xml)',
                        default='../data/Q1_sample.xml')
    parser.add_argument('--model-name',
                        dest='model_name',
                        help='Model file name',
                        default='Q1_model')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true')
    parser.add_argument('--workers',
                        dest='workers',
                        help='How many workers to train w2v model',
                        default=4,
                        type=int)
    parser.add_argument('--iterations',
                        dest='iterations',
                        help='How many training iterations',
                        default=10,
                        type=int)
    parser.add_argument('--window',
                        dest='window',
                        help='Window size (word2vec)',
                        default=5,
                        type=int)
    args = parser.parse_args()

    train_model(args.data, args.model_name, window=args.window,
                workers=args.workers, iterations=args.iterations, verbose=args.verbose)
