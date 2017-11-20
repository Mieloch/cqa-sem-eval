from basic_stats import load, remove_subject_from_question
from bs4 import BeautifulSoup
import word2vec_model.word2vec_utils as word2vec
import gensim
import numpy as np

def word2vec_dataset(xml_file, word2vec_model):
    print("Loading subtask C word2vec dataset")
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
    print("Loading subtask C word2vec dataset [DONE]")
    return np.asarray(questions), np.asarray(comments), np.asarray(relevace_labels)


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
            org_question_body = org_question.OrgQBody.text
            if org_question_body == "":
                print("WARN! empty question")
                continue
        rel_comments = org_question("RelComment")
        for rel_comment in rel_comments:
            rel_comment_text = rel_comment.RelCText.text
            if rel_comment_text == "":
                print("WARN! empty comment")
                continue
            orgq_relc_pair = dict([("orgq", org_question_body),
                                   ("relc", rel_comment_text),
                                   ("relc_orgq_relevance", rel_comment["RELC_RELEVANCE2ORGQ"])])
            dataset.append(orgq_relc_pair)
    print("Loading subtask C raw dataset [DONE]")
    return dataset


#model = word2vec.load_word2vec_model("SemEval2016-Task3-CQA-QL-dev_model")
#q, c, r = word2vec_dataset("../data/SemEval2016-Task3-CQA-QL-dev.xml", model)
#d = raw_dataset("data/Q1_sample.xml")
