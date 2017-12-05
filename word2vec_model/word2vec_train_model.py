import gensim
import basic_stats
from word2vec_utils import tokenize_to_lower_case
from lxml import etree

class Sentences(object):
    def __init__(self, xml_path, verbose=False):
        self.xml_path = xml_path
        self.tokens_count = 0
        self.verbose = verbose

    def __iter__(self):
        processed_ids = []
        self.tokens_count = 0

        for event, original_question in etree.iterparse(self.xml_path, tag="OrgQuestion"):
            id = original_question.get("ORGQ_ID")

            if id not in processed_ids:
                processed_ids.append(id)
                original_question_body = original_question.findtext("OrgQBody")
                original_question_body = basic_stats.remove_subject_from_question(
                    original_question_body
                )
                tokenize = tokenize_to_lower_case(original_question_body)
                self.tokens_count += len(tokenize)
                yield tokenize

            # related questions
            related_questions = original_question.findall(".//RelQuestion")
            for related_question in related_questions:
                related_question_body = related_question.findtext('RelQBody')
                related_question_body = basic_stats.remove_subject_from_question(
                    related_question_body)
                tokenize = tokenize_to_lower_case(related_question_body)
                self.tokens_count += len(tokenize)
                yield tokenize

            # related comments
            related_comments = original_question.findall(".//RelComment")
            for related_comment in related_comments:
                related_comment_body = related_comment.findtext('RelCText')
                tokenize = tokenize_to_lower_case(related_comment_body)
                # tokenize = tokenize_to_lower_case(related_comment.RelCClean.text)
                self.tokens_count += len(tokenize)
                yield tokenize

            # related answers (only for subtask e)
            related_answers = original_question.findall(".//RelAnswer")
            for related_answer in related_answers:
                related_answer_body = related_answer.findtext('RelAText')
                tokenize = tokenize_to_lower_case(related_answer_body)
                self.tokens_count += len(tokenize)
                yield tokenize

            if self.verbose:
                print("Tokens count = {}".format(self.tokens_count))


def train_model(data_src, model_name, verbose=False):
    sentences = Sentences(data_src, verbose=verbose)
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
