import sys
sys.path.append('../..')

import spacy
import basic_stats
import pandas as pd
from os import path
from lxml import etree
from word2vec_model import word2vec_utils

class StrippedQuestions(object):
    def __init__(xml_path, w2v_model=None, append=False, skip=0, verbose=False):
        self.xml_path = xml_path
        self.verbose = verbose
        self.w2v_model = w2v_model
        self.append = append
        self.skip = skip
        self.iteration = 0
        self.progress = 0.0

    def __iter__(self):
        df = pd.read_csv(self.xml_path)

        for question in questions:
            print(question)
