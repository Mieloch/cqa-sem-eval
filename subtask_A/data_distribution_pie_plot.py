import matplotlib.pyplot as plt
import pandas as pd

RELEVANCE = "relevance"


def plot_pie(data_set_file, title):
    data_set = pd.read_csv(data_set_file)
    relevance_good = len(data_set[data_set[RELEVANCE] == "Good"])
    relevance_bad = len(data_set[data_set[RELEVANCE] == "Bad"]) + len(
        data_set[data_set[RELEVANCE] == "PotentiallyUseful"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([relevance_good, relevance_bad], labels=['Good', 'Bad'], autopct='%1.3f%%',
            startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig.suptitle(title)
    plt.savefig("plots/{}.png".format(title))

plot_pie("csv/train_data_set.csv", "Train set class distribution")
plot_pie("csv/validation_data_set.csv", "Validation set class distribution")
