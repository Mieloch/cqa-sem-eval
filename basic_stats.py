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


def ngram_similarity(original, related, n=2):
    org_tokens = nltk.word_tokenize(original)
    rel_tokens = nltk.word_tokenize(related)
    org_terms = set(nltk.ngrams(org_tokens, n))
    rel_terms = set(nltk.ngrams(rel_tokens, n))

    shared_terms = org_terms.intersection(rel_terms)
    all_terms = org_terms.union(rel_terms)

    dist = 1.0  # if =1.0 -> no common terms
    if len(all_terms) > 0:
        dist = 1.0 - (len(shared_terms) / float(len(all_terms)))

    return dist
