import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
from scipy import stats
import seaborn as sns; sns.set()
sns.set(style="darkgrid")


with open('description_dump_finetuned.pickle', 'rb') as f:
    description, indices, answers, counts, accuracies, stabilities = pickle.load(f)

for experiment in description.keys():
    for polarity in description[experiment].keys():
        if polarity in ['accuracies', 'stabilities']:
            continue
        # stability etc
        if isinstance(description[experiment][polarity], float):
            continue

        for word_location in description[experiment][polarity].keys():
            description[experiment][polarity][word_location] = \
                np.array(description[experiment][polarity][word_location])

    answers[experiment] = np.array(answers[experiment])

dists = []
ratios = []

for experiment in description.keys():
    correct_original = description['text_original']['correct']['ans']
    wrong_original = description['text_original']['wrong']['ans']
    current_indices = indices[experiment]['ans']

    correct_shift = math.e ** description[experiment]['correct']['ans'] - math.e ** correct_original[current_indices]
    wrong_shift = math.e ** description[experiment]['wrong']['ans'] - math.e ** wrong_original[current_indices]

    current_answers = answers[experiment]
    # assert np.sum(current_answers) / len(current_answers) == description[experiment]['accuracy']

    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        experiment,
        100 * accuracies[experiment]['all'] / counts[experiment]['all'],
        100 * accuracies[experiment]['associative'] / counts[experiment]['associative'],
        100 * accuracies[experiment]['switchable'] / counts[experiment]['switchable'],
        100 * accuracies[experiment]['!associative'] / counts[experiment]['!associative'],
        100 * accuracies[experiment]['!switchable'] / counts[experiment]['!switchable'],
        100 * stabilities[experiment]['all'] / counts[experiment]['all'],
        100 * stabilities[experiment]['associative'] / counts[experiment]['associative'],
        100 * stabilities[experiment]['switchable'] / counts[experiment]['switchable'],
        100 * stabilities[experiment]['!associative'] / counts[experiment]['!associative'],
        100 * stabilities[experiment]['!switchable'] / counts[experiment]['!switchable'],
    ))
    # subset of answers for the original dataset that matches valid ones for current experiment
    subset_score = np.sum(answers['text_original'][current_indices]) / len(answers['text_original'][current_indices])

#     correct_dis = description[experiment]['correct']['dis']
#     correct_dis = correct_dis[correct_dis != None]
#
#     wrong_dis = description[experiment]['wrong']['dis']
#     wrong_dis = wrong_dis[wrong_dis != None]
#
#     ratios = (math.e **  correct_dis - math.e ** wrong_dis).astype(float)
#     #plt.ylim(0.3, 0.4)
#     # sns.barplot(list(range(len(ratios))), ratios)
#
#     # plt.show()
#     pdf = stats.norm.pdf(ratios)
#     sns.lineplot(ratios, pdf)
#
# plt.show()
#     print("{}\t{:.5f}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}".
#           format(experiment, np.mean(correct_shift), np.mean(wrong_shift),
#     100 * subset_score, 100 * description[experiment]['accuracy'], 100 * description[experiment]['stability']))

"""
plt.ylim(top=3)
plt.bar(indices, final_probs, 0.4, color='olivedrab')
plt.bar(indices + 0.4, final_enhanced, 0.4, color='darksalmon')
plt.show()
"""