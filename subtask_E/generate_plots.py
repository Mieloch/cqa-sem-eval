import matplotlib.pyplot as plt
import pandas as pd
import os

def create_plot(stats_path, title=None):
    if title is None:
        base = os.path.basename(stats_path)
        title = os.path.splitext(stats_path)[0]

    df = pd.read_csv(stats_path)

    # Load data for different
    irrelevant = df[df.relevance == 'Irrelevant']
    related = df[df.relevance == 'Related']
    perfect_match = df[df.relevance == 'PerfectMatch']

    assert(irrelevant.shape[0] + related.shape[0] +
           perfect_match.shape[0] == df.shape[0])

    labels = ['Irrelevant', 'Related', 'Perfect Match']
    parameters = ['jaccard_distance', 'length_difference', 'cosine_similarity',
                  'bigram_similarity', 'w2v_cosine_similarity', 'score',
                  'view_count', 'no_of_comments', 'no_of_answers',
                  'total_answer_upvotes', 'best_answer_upvotes']

    for i, param in enumerate(parameters):
        fig = plt.figure()
        plot_data = [irrelevant[param], related[param], perfect_match[param]]

        fig.suptitle('{} - {}'.format(title, param))
        plt.boxplot(plot_data)
        plt.xticks(list(range(1, len(labels) + 1)), labels)

        fig_name = 'plots/{}-{}.png'.format(title, param)
        fig.savefig(fig_name)

        plt.close(fig)

def generate_plots():
    df_paths = {'en_dev_100': '/Volumes/DataDrive/stats/stackexchange_english_devel.xml--stackexchange_english_devel.xml--it50-mc5-s100.mdl.csv',
                'en_dev_200': '/Volumes/DataDrive/stats/stackexchange_english_devel.xml--stackexchange_english_devel.xml--it50-mc5-s200.mdl.csv',
                'en_dev_gn': '/Volumes/DataDrive/stats/stackexchange_english_devel.xml--GoogleNews-vectors-negative300.bin.csv',
                'en_train_100': '/Volumes/DataDrive/stats/stackexchange_english_train.xml--stackexchange_english_devel.xml--it50-mc5-s100.mdl.csv',
                'en_train_200': '/Volumes/DataDrive/stats/stackexchange_english_train.xml--stackexchange_english_devel.xml--it50-mc5-s200.mdl.csv',
                'en_train_gn': '/Volumes/DataDrive/stats/stackexchange_english_train.xml--GoogleNews-vectors-negative300.bin.csv'}

    for title, path in df_paths.items():
        create_plot(path, title=title)



if __name__ == '__main__':
    generate_plots()
