from basic_stats import load
import word2vec_model.word2vec_utils as word2vec
import numpy as np
import pandas as pd
import string
from stop_words import get_stop_words
from itertools import filterfalse


GOOD = 0
POT_USEFUL = 1
BAD = 2


def raw_dataframe(xml_file):
    print("Loading subtask C raw dataset")
    soup = load(xml_file)
    org_questions = soup("OrgQuestion")
    processed_ids = []
    questions = []
    comments = []
    relevances = []
    for org_question in org_questions:
        id = org_question["ORGQ_ID"]
        if id not in processed_ids:
            processed_ids.append(id)
            org_question_body = org_question.OrgQBody.text
            if org_question_body == "":
                print("WARN! empty question")
                continue
            questions.append(org_question_body)
            threads_comments = []
            comments.append(threads_comments)
            threads_relevances = []
            relevances.append(threads_relevances)
        rel_comments = org_question("RelComment")
        for rel_comment in rel_comments:
            rel_comment_text = rel_comment.RelCText.text
            if rel_comment_text == "":
                print("WARN! empty comment")
                continue
            threads_comments.append(rel_comment_text)
            threads_relevances.append(rel_comment["RELC_RELEVANCE2ORGQ"])
    question_series = pd.Series(questions)
    comment_series = pd.Series(comments)
    relevance_series = pd.Series(relevances)
    data_set = pd.concat((
        pd.DataFrame(series) for series in (question_series, comment_series, relevance_series)),
        axis=1
    )
    data_set.columns = ["question", "comments", "relevance"]
    print("Loading subtask C raw dataset [DONE]")
    return data_set


def word_vecs_dataset(xml_file, word2vec_model):
    print("Loading subtask C vectors dataset")
    text_dataframe = raw_dataframe(xml_file)
    question_vectors_series = convert_questions(text_dataframe.question, word2vec_model)
    comment_vector_series = convert_comments(text_dataframe.comments, word2vec_model)
    relevance_onehot_series = convert_relevances(text_dataframe.relevance)
    vec_dataframe = pd.concat((pd.DataFrame(series) for series in (question_vectors_series, comment_vector_series, relevance_onehot_series)),
                              axis=1)
    vec_dataframe.columns = ["question", "comments", "relevance"]
    print("Loading subtask C vectors dataset [DONE]")
    return vec_dataframe


stop_words = get_stop_words('en')


def token_in_stop_words(token):
    return token in stop_words


def text_to_vector_series(text_series, word2vec_model):
    text_vecs_series = []
    for text in text_series:
        text_vectors = []
        text_vecs_series.append(text_vectors)
        text = word2vec.tokenize_to_lower_case(text)
        text[:] = filterfalse(token_in_stop_words, text)
        for word_token in text:
            word_vector = word2vec_model.wv[word_token]
            text_vectors.append(word_vector)
    return text_vecs_series


def convert_questions(question_series, word2vec_model):
    return pd.Series(text_to_vector_series(question_series, word2vec_model))


def convert_comments(comment_series, word2vec_model):
    comment_vecs_series = []
    for rel_comments in comment_series:
        comments_vecs = text_to_vector_series(rel_comments, word2vec_model)
        comment_vecs_series.append(comments_vecs)
    return pd.Series(comment_vecs_series)


def convert_relevances(relevance_series):
    onehot_relevances_series = []
    for relevances in relevance_series:
        onehot_relevances = []
        onehot_relevances_series.append(onehot_relevances)
        for relevance in relevances:
            if relevance == "Good":
                onehot_relevances.append([1, 0, 0])
            elif relevance == "PotentiallyUseful":
                onehot_relevances.append([0, 1, 0])
            elif relevance == "Bad":
                onehot_relevances.append([0, 0, 1])
    return pd.Series(onehot_relevances_series)


def raw_dataset(xml_file):
    print("Loading subtask C raw dataset")
    soup = load(xml_file)
    org_questions = soup("OrgQuestion")
    dataset = []
    processed_ids = []
    for org_question in org_questions:
        id = org_question["ORGQ_ID"]
        if id not in processed_ids:
            processed_ids.append(id)
            org_question_body = filter_latin_alphabet(bytearray(org_question.OrgQBody.text, "utf-8"))
            if org_question_body == "":
                print("WARN! empty question")
                continue
        rel_comments = org_question("RelComment")
        for rel_comment in rel_comments:
            rel_comment_text = filter_latin_alphabet(bytearray(rel_comment.RelCText.text, "utf-8"))
            if rel_comment_text == "":
                print("WARN! empty comment")
                continue
            orgq_relc_pair = dict([("orgq", org_question_body),
                                   ("relc", rel_comment_text),
                                   ("relc_orgq_relevance", get_comment_relevance(rel_comment))])
            dataset.append(orgq_relc_pair)
    print("Loading subtask C raw dataset [DONE]")
    return dataset


def sentences_dataset(xml_file, word2vec_model):
    print("Loading subtask C sentences dataset")
    raw_dataset_dict = raw_dataset(xml_file)
    questions = []
    comments = []
    relevace_labels = []
    for sample in raw_dataset_dict:
        orgq_vector = word2vec.sentence_vectors_mean(
            word2vec.sentence2vectors(sample["orgq"], word2vec_model, exclude_stopwords=True, to_lower_case=True))
        relc_vector = word2vec.sentence_vectors_mean(
            word2vec.sentence2vectors(sample["relc"], word2vec_model, to_lower_case=True, exclude_stopwords=True))
        if len(orgq_vector) == 0 or len(relc_vector) == 0:
            continue
        questions.append(orgq_vector)
        comments.append(relc_vector)
        relevace_labels.append(sample["relc_orgq_relevance"])
    print("Loading subtask C sentences dataset [DONE]")
    return np.asarray(questions), np.asarray(comments), np.asarray(relevace_labels)


def get_comment_relevance(rel_comment):
    relevance = rel_comment["RELC_RELEVANCE2ORGQ"]
    if relevance == "Good":
        return np.array(GOOD)
    elif relevance == "PotentiallyUseful":
        return np.array(POT_USEFUL)
    elif relevance == "Bad":
        return np.array(BAD)


def filter_latin_alphabet(in_text):
    out_text = ""
    for char in in_text.decode("utf-8"):
        if char in string.ascii_letters or char == " ":
            out_text += char
    return out_text