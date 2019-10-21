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

with open('description_dump_bert_headmaxnonorm_pronmean.pickle', 'rb') as f:
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

correct_diffs_flat = []
wrong_diffs_flat  = []
correct_shifts_flat  = []
wrong_shifts_flat  = []


all_diffs = []
all_shifts = []

all_diffs_flat = []
all_shifts_flat = []

all_accuracies = []
pron_diffs = []

pron_diffs_flat = []


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

    pron_attn_orig = description['text_original']['all']['pron_attn'][current_indices]
    pron_attn = description[experiment]['all']['pron_attn']
    pron_diff = pron_attn - pron_attn_orig


    print("{}\t correct_shift: {}\t wrong_shift: {}\t corr diff: {}\t wrong diff: {}\t acc: {}".format(experiment, correct_shift.mean(), wrong_shift.mean(), correct_diff.mean(), wrong_diff.mean(), accuracies[experiment]['all'] / len(current_indices)))
    #diffs
    correct_diffs.append(correct_diff.mean())
    wrong_diffs.append(wrong_diff.mean())
    pron_diffs.append(pron_diff.mean())
    all_diffs.append(correct_diff.mean() + wrong_diff.mean())

    #flat
    correct_diffs_flat.extend(correct_diff)
    wrong_diffs_flat.extend(wrong_diff)
    pron_diffs_flat.extend(pron_diff)
    all_diffs_flat.extend([c + w for c, w in zip(correct_diff , wrong_diff)])

    #shifts
    correct_shifts.append(correct_shift.mean())
    wrong_shifts.append(wrong_shift.mean())
    all_shifts.append(correct_shift.mean() + wrong_shift.mean())

    #flat
    correct_shifts_flat.extend(correct_shift)
    wrong_shifts_flat.extend(wrong_shift)
    all_shifts_flat.extend([c + w for c, w in zip(correct_shift , wrong_shift)])





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


#drop freq noun
correct_diffs.pop(-2)
correct_shifts.pop(-2)
all_accuracies.pop(-2)
wrong_diffs.pop(-2)
wrong_shifts.pop(-2)
pron_diffs.pop(-2)
all_shifts.pop(-2)


correct_diffs.pop(0)
correct_shifts.pop(0)
all_accuracies.pop(0)
wrong_diffs.pop(0)
wrong_shifts.pop(0)
pron_diffs.pop(0)
all_shifts.pop(0)

shift_diff = [c - w for c, w in zip(correct_shifts , wrong_shifts)]
print(stats.pearsonr(all_shifts, all_accuracies))
"""
plt.ylim(top=3)
plt.bar(indices, final_probs, 0.4, color='olivedrab')
plt.bar(indices + 0.4, final_enhanced, 0.4, color='darksalmon')
plt.show()
"""