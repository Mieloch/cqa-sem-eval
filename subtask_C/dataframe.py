from basic_stats import load
import pandas as pd
from itertools import filterfalse
from stop_words import get_stop_words
import word2vec_model.word2vec_utils as word2vec


def get_dataset(xml_file, model_name):
    org_qs, rel_cs, rels = get_subtask_C_data_from_file(xml_file, model_name)
    questions = pd.Series(org_qs)
    comments = pd.Series(rel_cs)
    relevances = pd.Series(rels)
    dataset = pd.concat((pd.DataFrame(series) for series in (questions, comments, relevances)), axis=1)
    dataset.columns = ["question", "comment", "relevance"]
    return dataset


def get_subtask_C_data_from_file(xml_file, model_name):
    soup = load(xml_file)
    org_questions = soup("OrgQuestion")
    questions = []
    comments = []
    relevances = []
    word2vec_model = word2vec.load_word2vec_model(model_name)
    for org_question in org_questions:
        rel_comments = org_question("RelComment")
        for rel_comment in rel_comments:
            questions.append(text_to_word_vectors(org_question.OrgQBody.text, word2vec_model))
            comments.append(text_to_word_vectors(rel_comment.RelCText.text, word2vec_model))
            relevances.append(get_comment_relevance(rel_comment))
    return questions, comments, relevances


def text_to_word_vectors(text, word2vec_model):
    vector_list = []
    tokens = word2vec.tokenize_to_lower_case(text)
    tokens[:] = filterfalse(token_in_stop_words, tokens)
    for token in tokens:
        word_vector = word2vec_model.wv[token]
        vector_list.append(word_vector)
    return vector_list


stop_words = get_stop_words('en')


def token_in_stop_words(token):
    return token in stop_words


def get_comment_relevance(rel_comment):
    relevance = rel_comment["RELC_RELEVANCE2ORGQ"]
    if relevance == "Good":
        return [1, 0, 0]
    elif relevance == "PotentiallyUseful":
        return [0, 1, 0]
    elif relevance == "Bad":
        return [0, 0, 1]


# data = get_dataset("data\SemEval2016-Task3-CQA-QL-dev.xml", "SemEval2016-Task3-CQA-QL-dev_model")
# print(data.relevance)