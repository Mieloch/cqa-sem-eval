from subtask_C import dataframe
import pandas as pd


def count_relevance(rels, pattern):
    matched_count = 0
    for rel in rels:
        if rel == pattern:
            matched_count += 1
    return matched_count


TRAIN_CSV = 'subtask_C\\csv_data\\data_not_binarized\\train.csv'
train_df = pd.read_csv(TRAIN_CSV)

s = train_df["relc_orgq_relevance"]

total = len(s)
good = count_relevance(s, 2)
potentially_useful = count_relevance(s, 1)
bad = count_relevance(s, 0)

good_overall = good + potentially_useful

good_percent = good / total * 100
potentially_useful_percent = potentially_useful / total * 100
bad_percent = bad / total * 100
good_overall_percent = good_overall / total * 100

print("Good %: " + str(good_percent))
print("PotentiallyUseful %: " + str(potentially_useful_percent))
print("Bad %: " + str(bad_percent))

print("Good overall %: " + str(good_overall_percent))