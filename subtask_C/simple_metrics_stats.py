import matplotlib.pyplot as plt
import pandas as pd

RELEVANCE = "relevance"
LENGTH_DIFFERENCE = "length_difference"
JACCARD_DISTANCE = "jaccard_distance"
COSINE_SIMILARITY = "cosine_similarity"
STATS_FILE = "subtask_C\OrgQuestion_to_RelComment_stats_binary_relevance.csv"

def plot_stat(stat, title):
    data_set = pd.read_csv(STATS_FILE)
    relevance_good = data_set[data_set[RELEVANCE] == 1][[stat]].values
    relevance_bad = data_set[data_set[RELEVANCE] == 0][[stat]].values

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
    axes[0].boxplot(relevance_good, showfliers=False)
    axes[0].set_title('Relevance: Good')
    axes[0].get_xaxis().set_visible(False)

    axes[1].boxplot(relevance_bad, showfliers=False)
    axes[1].set_title('Relevance: Bad')
    axes[1].get_xaxis().set_visible(False)

    fig.suptitle(title)
    if stat == LENGTH_DIFFERENCE:
        plt.ylim(ymin=-100)

    plt.savefig("subtask_C\plots\{}.png".format(title))


plot_stat(LENGTH_DIFFERENCE, "Length difference")
plot_stat(COSINE_SIMILARITY, "Cosine similarity")
plot_stat(JACCARD_DISTANCE, "Jaccard distance")