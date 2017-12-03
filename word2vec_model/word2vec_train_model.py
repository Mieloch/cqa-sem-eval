import gensim
import basic_stats
from word2vec_utils import tokenize_to_lower_case


def train_model(data_src, model_name, verbose=False):
    soup = basic_stats.load(data_src)
    original_questions = soup.findAll("OrgQuestion")

    sentences = []
    processed_ids = []
    tokens_count = 0

    def parse_original_question(question_element):
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
        related_answers = original_question.findAll("RelAnswer")
        for related_answer in related_answers:
            body = related_answer.RelAText.text
            tokenize = tokenize_to_lower_case(body)
            tokens_count += len(tokenize)
            sentences.append(tokenize)

    # creating corpus
    for original_question in original_questions:
        # original questions
        parse_original_question(original_question)

    if verbose:
        print("Token count: {}".format(tokens_count))

    model = gensim.models.Word2Vec(
        sentences, min_count=1, window=3, iter=1000, size=100)
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
    args = parser.parse_args()

    train_model(args.data, args.model_name, verbose=args.verbose)
