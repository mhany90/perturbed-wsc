import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
from scipy import stats
from functools import reduce
import seaborn as sns; sns.set()
sns.set(style="darkgrid")

with open('description_dump_pmi_log.pickle', 'rb') as f:
    description, indices, answers, counts, accuracies, stabilities, all_pmi_diffs = pickle.load(f)

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

all_accuracies = []
pron_diffs = []

pron_diffs_flat = []


for experiment in description.keys():
    correct_original = description['text_original']['correct']['ans']
    wrong_original = description['text_original']['wrong']['ans']
    current_indices = indices[experiment]['ans']

    all_accuracies.append(accuracies[experiment]['all'] / len(current_indices))
    print_raw = False
    if print_raw:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
             experiment,
             accuracies[experiment]['all'],
             accuracies[experiment]['associative'],
             accuracies[experiment]['switchable'],
             accuracies[experiment]['!associative'],
             accuracies[experiment]['!switchable'],
             stabilities[experiment]['all'],
             stabilities[experiment]['associative'],
             stabilities[experiment]['switchable'],
             stabilities[experiment]['!associative'],
             stabilities[experiment]['!switchable'],
             counts[experiment]['all'],
             counts[experiment]['associative'],
             counts[experiment]['switchable'],
             counts[experiment]['!associative'],
             counts[experiment]['!switchable'],
        ))
    else:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            experiment,
            accuracies[experiment]['all'] / counts[experiment]['all'],
            accuracies[experiment]['associative'] / counts[experiment]['associative'],
            accuracies[experiment]['switchable'] /  counts[experiment]['switchable'],
            accuracies[experiment]['!associative'] / counts[experiment]['!associative'],
            accuracies[experiment]['!switchable'] / counts[experiment]['!switchable'],
            stabilities[experiment]['all']  / counts[experiment]['all'],
            stabilities[experiment]['associative']  / counts[experiment]['associative'],
            stabilities[experiment]['switchable'] / counts[experiment]['switchable'],
            stabilities[experiment]['!associative']  / counts[experiment]['!associative'],
            stabilities[experiment]['!switchable'] / counts[experiment]['!switchable'],
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


#drop freq noun

#all_accuracies.pop(-2)
#all_accuracies.pop(0)


"""
plt.ylim(top=3)
plt.bar(indices, final_probs, 0.4, color='olivedrab')
plt.bar(indices + 0.4, final_enhanced, 0.4, color='darksalmon')
plt.show()
"""