import pandas as pd
from shutil import copyfile

RELEVANCE = "relevance"
STATS_FILE = "OrgQuestion_to_RelComment_stats.csv"
STATS_FILE_BINARY = "OrgQuestion_to_RelComment_stats_binary_relevance.csv"

fieldnames = ['question1_id', 'related_comment_id', 'jaccard_distance', 'length_difference',
                  'cosine_similarity', 'bigram_similarity', 'w2v_cosine_similarity', 'relevance']

copyfile(STATS_FILE, STATS_FILE_BINARY)
data_set = pd.read_csv(STATS_FILE_BINARY)
data_set[RELEVANCE] = data_set[RELEVANCE].map({'PotentiallyUseful': 1, 'Good': 1, 'Bad': 0})
data_set.to_csv(STATS_FILE_BINARY, columns=fieldnames, index=False, float_format='%.3f')
