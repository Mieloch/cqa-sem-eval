from basic_stats import remove_subject_from_question, load
from bs4 import BeautifulSoup
import word2vec_model.word2vec_utils as word2vec
import gensim


def subtask_A_word2vec_dataset(xml_file, word2vec_model):
    print("Loading word2vec data set")
    data_set = subtask_A_raw_dataset(xml_file)
    result = []
    for sample in data_set:
        question_vector = word2vec.sentence_vectors_mean(
            word2vec.sentence2vectors(sample["question"], word2vec_model, exclude_stopwords=True, to_lower_case=True))
        comment_vector = word2vec.sentence_vectors_mean(
            word2vec.sentence2vectors(sample["comment"], word2vec_model, to_lower_case=True, exclude_stopwords=True))
        if len(question_vector) == 0 or len(comment_vector) == 0:
            continue
        transformed_sample = dict([
            ("question", question_vector),
            ("comment", comment_vector),
            ("relevance", label_to_class(sample["relevance"]))])
        result.append(transformed_sample)
    print("Loading word2vec data set [DONE]")
    return result


def subtask_A_raw_dataset(xml_file):
    print("Loading raw data set")
    soup = load(xml_file)
    threads = soup.findAll('Thread', recursive=True)
    # print(len(threads))
    data_set = []
    for thread in threads:
        thread_soup = BeautifulSoup(str(thread), "xml")
        question = thread_soup.RelQuestion.RelQBody.text
        if question == '':
            print("WARN! empty question")
            continue
        comments = thread_soup.findAll("RelComment")
        # print(len(comments))
        for comment in comments:
            comment_text = comment.RelCText.text
            if comment_text == '':
                print("WARN! empty comment")
                continue
            data_set_sample = dict([("question", remove_subject_from_question(question)),
                                    ("comment", comment_text),
                                    ("relevance", comment['RELC_RELEVANCE2RELQ'])])
            # print(data_set_sample)
            data_set.append(data_set_sample)
    print("Loading raw data set [DONE]")
    return data_set


def label_to_class(label):
    if label == "Good":
        return 0
    if label == "PotentiallyUseful":
        return 1
    if label == "Bad":
        return 2

# t = subtask_A_dataset("../data/Q1_sample.xml")
# print(t)
