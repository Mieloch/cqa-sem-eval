import sys
sys.path.append('..')

import basic_stats
import gensim
import csv
import spacy
import signal
from word2vec_model import word2vec_utils
from lxml import etree
from os import path

ORIGINAL_QUESTION_ID = "question_id"
RELATED_ID = "related_id"
RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
BIGRAM_SIMILARITY = "bigram_similarity"
W2V_COSINE_SIMILARITY = "w2v_cosine_similarity"
TYPE='type'

def kill_program(output_file, iteration_num):
    print("Program stopped at {} file, {} iteration".format(output_file, iteration_num))
    exit()

class SigTermKiller:
    def __init__(self):
        self.kill = False
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        self.kill = True


class Questions(object):
    def __init__(self, xml_path, w2v_model=None, append=False, skip=0, verbose=False):
        self.xml_path = xml_path
        self.verbose = verbose
        self.w2v_model = w2v_model
        self.append = append
        self.skip = skip
        self.iteration = 0
        pass

    def related_question(self, original_question, model):
        original_question_body = original_question.findtext("OrgQBody")
        related_question = original_question.find(".//RelQuestion")
        related_question_body = related_question.findtext("RelQBody")

        row = {}
        row[ORIGINAL_QUESTION_ID] = original_question.get('ORGQ_ID')
        row[TYPE] = 'RELQ'
        row[RELATED_ID] = related_question.get('RELQ_ID')
        row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(
            original_question_body, related_question_body), 3)
        row[LENGTH_DIFFERENCE] = basic_stats.length_difference(
            original_question_body, related_question_body)
        row[COSINE_SIMILARITY] = round(basic_stats.cosine_similarity(
            model, original_question_body, related_question_body), 3)
        row[BIGRAM_SIMILARITY] = round(basic_stats.ngram_similarity(
            original_question_body, related_question_body, n=2), 3)

        if self.w2v_model:
            related_question_vector = word2vec_utils.sentence_vectors_mean(
                word2vec_utils.sentence2vectors(related_question_body,
                                                self.w2v_model,
                                                to_lower_case=True,
                                                verbose=False))
            original_question_vector = word2vec_utils.sentence_vectors_mean(
                word2vec_utils.sentence2vectors(original_question_body,
                                                self.w2v_model,
                                                to_lower_case=True,
                                                verbose=False))

            row[W2V_COSINE_SIMILARITY] = word2vec_utils.vectors_cosine_similarity(original_question_vector,
                                                                                  related_question_vector)

        row[RELEVANCE] = related_question.get('RELQ_RELEVANCE2ORGQ')
        return row

    def related_answers(self, original_question, model):
        original_question_body = original_question.findtext("OrgQBody")
        related_answers = original_question.findall(".//RelAnswer")

        answers = []
        for related_answer in related_answers:
            related_answer_body = related_answer.findtext("RelAText")

            row = {}
            row[ORIGINAL_QUESTION_ID] = original_question.get('ORGQ_ID')
            row[TYPE] = 'RELA'
            row[RELATED_ID] = related_answer.get('RELA_ID')
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(
                original_question_body, related_answer_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(
                original_question_body, related_answer_body)
            row[COSINE_SIMILARITY] = round(basic_stats.cosine_similarity(
                model, original_question_body, related_answer_body), 3)
            row[BIGRAM_SIMILARITY] = round(basic_stats.ngram_similarity(
                original_question_body, related_answer_body, n=2), 3)

            if self.w2v_model:
                related_answer_vector = word2vec_utils.sentence_vectors_mean(
                    word2vec_utils.sentence2vectors(related_answer_body,
                                                    self.w2v_model,
                                                    to_lower_case=True,
                                                    verbose=False))
                original_question_vector = word2vec_utils.sentence_vectors_mean(
                    word2vec_utils.sentence2vectors(original_question_body,
                                                    self.w2v_model,
                                                    to_lower_case=True,
                                                    verbose=False))

                row[W2V_COSINE_SIMILARITY] = word2vec_utils.vectors_cosine_similarity(original_question_vector,
                                                                                    related_answer_vector)

            answers.append(row)

        return answers

    def related_comments(self, original_question, model):
        original_question_body = original_question.findtext("OrgQBody")
        related_comments = original_question.findall(".//RelComment")

        comments = []
        for related_comment in related_comments:
            related_comment_body = related_comment.findtext("RelCText")

            row = {}
            row[ORIGINAL_QUESTION_ID] = original_question.get('ORGQ_ID')
            row[TYPE] = 'RELC'
            row[RELATED_ID] = related_comment.get('RELC_ID')
            row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(
                original_question_body, related_comment_body), 3)
            row[LENGTH_DIFFERENCE] = basic_stats.length_difference(
                original_question_body, related_comment_body)
            row[COSINE_SIMILARITY] = round(basic_stats.cosine_similarity(
                model, original_question_body, related_comment_body), 3)
            row[BIGRAM_SIMILARITY] = round(basic_stats.ngram_similarity(
                original_question_body, related_comment_body, n=2), 3)

            if self.w2v_model:
                related_comment_vector = word2vec_utils.sentence_vectors_mean(
                    word2vec_utils.sentence2vectors(related_comment_body,
                                                    self.w2v_model,
                                                    to_lower_case=True,
                                                    verbose=False))
                original_question_vector = word2vec_utils.sentence_vectors_mean(
                    word2vec_utils.sentence2vectors(original_question_body,
                                                    self.w2v_model,
                                                    to_lower_case=True,
                                                    verbose=False))

                row[W2V_COSINE_SIMILARITY] = word2vec_utils.vectors_cosine_similarity(original_question_vector,
                                                                                      related_comment_vector)

            comments.append(row)

        return comments


    def __iter__(self):
        model = spacy.load("en")
        file_size = path.getsize(self.xml_path)
        with open(self.xml_path, 'rb') as fp:
            for i, (event, original_question) in enumerate(etree.iterparse(fp, tag="OrgQuestion")):
                if self.skip > i:
                    continue

                self.iteration = i
                original_question_body = original_question.findtext("OrgQBody")

                # There's always one related question
                yield self.related_question(original_question, model)

                for answer_row in self.related_answers(original_question, model):
                    yield answer_row

                for comment_row in self.related_comments(original_question, model):
                    yield comment_row

                if self.verbose:
                    progress = float(fp.tell() / file_size) * 100.0
                    print("Iteration = {}, File progress = {:2.2f}%".format(i, progress), end="\r")


def stats(src, dest='Duplicate-Question-stats.csv', model_path='../word2vec_model/subtask-e-word-model', append=False, skip=0, verbose=False):
    if not path.isfile(src):
        raise FileNotFoundError("Source file not found at {}".format(src))

    if not path.isfile(model_path):
        raise FileNotFoundError("Model file not found at {}".format(model_path))

    sig_term_killer = SigTermKiller()

    word2vec_model = gensim.models.Word2Vec.load(model_path)

    file_mode = 'a' if append else 'w'
    with open(dest, file_mode) as csvfile:
        fieldnames = [ORIGINAL_QUESTION_ID, TYPE, RELATED_ID, JACCARD_DISTANCE,
                      LENGTH_DIFFERENCE, COSINE_SIMILARITY, BIGRAM_SIMILARITY, W2V_COSINE_SIMILARITY, RELEVANCE]
        writer = csv.DictWriter(
            csvfile, fieldnames=fieldnames, lineterminator='\n')

        if not append:
            writer.writeheader()

        questions_iterator = Questions(
            src, word2vec_model, append=append, skip=skip, verbose=verbose)
        for row in questions_iterator:
            writer.writerow(row)

            if sig_term_killer.kill:
                kill_program(dest, questions_iterator.iteration)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', help='Data file',
                        default='../data/subtask_e_sample.xml')
    parser.add_argument('--output', dest='output', help='Output stats .csv file',
                        default='Duplicate-Question-stats.csv')
    parser.add_argument('--model', dest='model', help='Word2Vec model file',
                        default='../word2vec_model/subtask-e-word-model')
    parser.add_argument('--skip', dest='skip', help='Word2Vec model file',
                        type=int, default=0)

    args = parser.parse_args()
    stats(args.data,
          dest=args.output,
          model_path=args.model,
          append=args.skip > 0,
          skip=args.skip,
          verbose=False)
