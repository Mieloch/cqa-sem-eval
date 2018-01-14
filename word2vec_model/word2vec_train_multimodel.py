import gensim
import basic_stats
from word2vec_utils import tokenize_to_lower_case
from lxml import etree
from collections import namedtuple
import os


class Sentences(object):
    def __init__(self, xml_path, verbose=False):
        self.xml_path = xml_path
        self.tokens_count = 0
        self.verbose = verbose

    def __iter__(self):
        processed_ids = []
        self.tokens_count = 0

        file_size = os.path.getsize(self.xml_path)
        with open(self.xml_path, mode='rb') as fp:
            for event, original_question in etree.iterparse(fp, tag="OrgQuestion"):
                id = original_question.get("ORGQ_ID")

                if id not in processed_ids:
                    processed_ids.append(id)
                    original_question_body = original_question.findtext(
                        "OrgQBody")
                    original_question_body = basic_stats.remove_subject_from_question(
                        original_question_body
                    )
                    tokenize = tokenize_to_lower_case(original_question_body)
                    self.tokens_count += len(tokenize)
                    yield tokenize

                # related questions
                related_questions = original_question.findall(".//RelQuestion")
                for related_question in related_questions:
                    related_question_body = related_question.findtext(
                        'RelQBody')
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
                    print("Tokens = {}, Progress = {}".format(
                        self.tokens_count,
                        float(fp.tell()) / file_size), end='\r')

                original_question.clear()

ModelDef = namedtuple('ModelDef', 'name, min_count, size, iterations')

def train_models(data_src, workers=4, verbose=False):
    sentences = list(Sentences(data_src, verbose=verbose))

    models = [
        ModelDef('it10-mc5-s100', 5, 100, 10),
        ModelDef('it30-mc5-s100', 5, 100, 30),
        ModelDef('it50-mc5-s100', 5, 100, 50),
        ModelDef('it10-mc10-s100', 10, 100, 10),
        ModelDef('it30-mc10-s100', 10, 100, 30),
        ModelDef('it50-mc10-s100', 10, 100, 50),
        ModelDef('it10-mc5-s200', 5, 200, 10),
        ModelDef('it30-mc5-s200', 5, 200, 30),
        ModelDef('it50-mc5-s200', 5, 200, 50),
        ModelDef('it10-mc10-s200', 10, 200, 10),
        ModelDef('it30-mc10-s200', 10, 200, 30),
        ModelDef('it50-mc10-s200', 10, 200, 50),
    ]

    for (name, min_count, size, iterations) in models:
        if verbose:
            print("Running \"{}\" model with min_count={}, iterations={}, size={}".format(
                name, min_count, iterations, size))

        # Create model name
        file_name = os.path.basename(data_src)
        model_name = "models/{}--{}.mdl".format(file_name, name)

        # Train model
        model = gensim.models.Word2Vec(
            sentences, min_count=min_count, iter=iterations, size=size, workers=workers)
        model.save(model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        dest='data',
                        help='Training data path (default = data/Q1_sample.xml)',
                        default='../data/Q1_sample.xml')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true')
    parser.add_argument('--workers',
                        dest='workers',
                        help='How many workers to train w2v model',
                        default=4,
                        type=int)
    args = parser.parse_args()

    train_models(args.data, workers=args.workers, verbose=args.verbose)
