import sys
sys.path.append('../..')

import spacy
import basic_stats
from os import path
from lxml import etree
from word2vec_model import word2vec_utils


class Questions(object):
    def __init__(self, xml_path, w2v_model=None, append=False, skip=0, verbose=False):
        self.xml_path = xml_path
        self.verbose = verbose
        self.w2v_model = w2v_model
        self.append = append
        self.skip = skip
        self.iteration = 0
        self.progress = 0.0

    def iterator_cleanup(self, element):
        element.clear()

        while element.getprevious() is not None:
            del element.getparent()[0]

    def __iter__(self):
        model = spacy.load("en_core_web_lg")
        file_size = path.getsize(self.xml_path)

        with open(self.xml_path, 'rb') as fp:
            for i, (event, original_question) in enumerate(etree.iterparse(fp, tag="OrgQuestion")):
                self.iteration = i

                if self.skip > i:
                    self.iterator_cleanup(original_question)
                    continue

                original_question_body = original_question.findtext("OrgQBody")
                related_question = original_question.find(".//RelQuestion")
                related_question_body = related_question.findtext('RelQBody')
                comments = original_question.findall('.//RelComment')
                answers = original_question.findall('.//RelAnswer')

                row = {}
                row[ORIGINAL_QUESTION_ID] = original_question.get('ORGQ_ID')
                row[RELATED_QUESTION_ID] = related_question.get('RELQ_ID')
                row[CATEGORY] = related_question.get('RELQ_CATEGORY')
                row[SCORE] = related_question.get('RELQ_SCORE')
                row[TAGS] = related_question.get('RELQ_TAGS').replace(',', ';')
                row[VIEW_COUNT] = related_question.get('RELQ_VIEWCOUNT')
                row[USER_ID] = related_question.get('RELQ_USERID')
                row[RELEVANCE] = related_question.get('RELQ_RELEVANCE2ORGQ')
                row[NO_OF_COMMENTS] = len(comments)
                row[NO_OF_ANSWERS] = len(answers)

                answers_scores = [int(answer.get('RELA_SCORE'))
                                  for answer in answers]
                row[TOTAL_ANSWER_UPVOTES] = sum(answers_scores)
                row[BEST_ANSWER_UPVOTES] = max(answers_scores, default=0)
                row[HAS_ACCEPTED_ANSWER] = any(
                    [bool(answer.get('RELA_ACCEPTED')) for answer in answers])

                row[JACCARD_DISTANCE] = round(basic_stats.jaccard_distance(
                    original_question_body, related_question_body), 3)
                row[LENGTH_DIFFERENCE] = basic_stats.length_difference(
                    original_question_body, related_question_body)
                row[COSINE_SIMILARITY] = round(basic_stats.cosine_similarity(
                    model, original_question_body, related_question_body), 3)
                row[BIGRAM_SIMILARITY] = round(basic_stats.ngram_similarity(
                    original_question_body, related_question_body, n=2), 3)

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

                if self.verbose:
                    self.progress = float(fp.tell() / file_size) * 100.0
                    print("Iteration = {}, File progress = {:2.2f}%".format(
                        i, self.progress), end="\r")

                yield row
                self.iterator_cleanup(original_question)
