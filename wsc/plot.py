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

with open('description_dump_bert_headmaxnonorm.pickle', 'rb') as f:
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
correct_diffs = []
wrong_diffs = []
correct_shifts = []
wrong_shifts = []

all_diffs = []
all_shifts = []
all_accuracies = []

for experiment in description.keys():
    correct_original = description['text_original']['correct']['ans']
    wrong_original = description['text_original']['wrong']['ans']
    current_indices = indices[experiment]['ans']

    correct_shift = math.e ** description[experiment]['correct']['ans'] - math.e ** correct_original[current_indices]
    wrong_shift = math.e ** description[experiment]['wrong']['ans'] - math.e ** wrong_original[current_indices]

    current_answers = answers[experiment]
    # assert np.sum(current_answers) / len(current_answers) == description[experiment]['accuracy']

    #attn
    correct_attn_orig = description['text_original']['correct']['attn'][current_indices]
    wrong_attn_orig = description['text_original']['wrong']['attn'][current_indices]

    correct_attn = description[experiment]['correct']['attn']
    correct_diff = correct_attn - correct_attn_orig

    wrong_attn = description[experiment]['wrong']['attn']
    wrong_diff = wrong_attn - wrong_attn_orig


    print("{}\t corr diff: {}\t wrong diff: {}\t acc: {}".format(experiment, correct_diff.mean(), wrong_diff.mean(), accuracies[experiment]['all'] / len(current_indices)))
    correct_diffs.append(correct_diff.mean())
    wrong_diffs.append(wrong_diff.mean())
    all_diffs.append(correct_diff.mean() + wrong_diff.mean())

    correct_shifts.append(correct_shift.mean())
    wrong_shifts.append(wrong_shift.mean())
    all_shifts.append(correct_shift.mean() + wrong_shift.mean())



    all_accuracies.append(accuracies[experiment]['all'] / len(current_indices))

    # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
    #     experiment,
    #     accuracies[experiment]['all'],
    #     accuracies[experiment]['associative'],
    #     accuracies[experiment]['switchable'],
    #     accuracies[experiment]['!associative'],
    #     accuracies[experiment]['!switchable'],
    #     stabilities[experiment]['all'],
    #     stabilities[experiment]['associative'],
    #     stabilities[experiment]['switchable'],
    #     stabilities[experiment]['!associative'],
    #     stabilities[experiment]['!switchable'],
    #     counts[experiment]['all'],
    #     counts[experiment]['associative'],
    #     counts[experiment]['switchable'],
    #     counts[experiment]['!associative'],
    #     counts[experiment]['!switchable']
    # ))
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

print(stats.pearsonr(correct_diffs, correct_shifts))
"""
plt.ylim(top=3)
plt.bar(indices, final_probs, 0.4, color='olivedrab')
plt.bar(indices + 0.4, final_enhanced, 0.4, color='darksalmon')
plt.show()
"""