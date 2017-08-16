from bs4 import BeautifulSoup
import re
import nltk


def load(file_name):
    with open(file_name, 'r') as myfile:
        return BeautifulSoup(myfile.read(), "xml")


def remove_subject_from_question(question):
    return re.sub('.+\/\/\ ', '', question)


def length_difference(original, related):
    return abs(len(original) - len(related))


def jaccard_distance(original, related):
    org_tokens = set(nltk.word_tokenize(original))
    rel_tokens = set(nltk.word_tokenize(related))

    return nltk.jaccard_distance(org_tokens, rel_tokens)


def cosine_similarity(model, original, related):
    org_tokens = model(original)
    rel_tokens = model(related)
    return org_tokens.similarity(rel_tokens)
