import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np
from scipy import stats
from functools import reduce
import pandas as pd
from collections import defaultdict


import seaborn as sns; sns.set()
sns.set(style="darkgrid")

with open('description_dump_bert_headmaxnonorm_pronmean.pickle', 'rb') as f:
    description, indices, answers, counts, accuracies, stabilities = pickle.load(f)

path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')
margin_ex_dir = 'margin_examples'

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
    wsc_datapoints_new =  wsc_datapoints[experiment]
    wsc_datapoints_new = [w for w in wsc_datapoints_new if w != '-']

    filtered_dps =  wsc_datapoints.loc[wsc_datapoints[experiment] != '-']
    wsc_pair_ids = filtered_dps['pair_number']

    correct_original = description['text_original']['correct']['ans']
    wrong_original = description['text_original']['wrong']['ans']
    current_indices = indices[experiment]['ans']

    prob_diffs = math.e ** description[experiment]['correct']['ans'] - math.e ** description[experiment]['wrong']['ans']
    prob_diffs_abs =  [abs(number) for number in prob_diffs]

    wsc_index_prob_diffs = {k:v for k,v in enumerate(prob_diffs)}
    wsc_index_prob_diffs_abs = {k:v for k,v in enumerate(prob_diffs_abs)}

    sorted_wsc_index_prob_diffs = sorted(wsc_index_prob_diffs.items(), key=lambda x: x[1], reverse=True)
    sorted_wsc_index_prob_diffs_abs = sorted(wsc_index_prob_diffs_abs.items(), key=lambda x: x[1])

    #top/bottom N instances
    top50 = sorted_wsc_index_prob_diffs[0:50]
    bottom50 = sorted_wsc_index_prob_diffs[-50:]
    top50idx = [idx_prob[0] for idx_prob in top50]
    bottom50idx = [idx_prob[0] for idx_prob in bottom50]

    top50_abs = sorted_wsc_index_prob_diffs_abs[0:50]
    top50idx_abs = [idx_prob[0] for idx_prob in top50_abs]

    small_pair_ids = []
    top_pair_ids = []
    bottom_pair_ids = []

    print('\n')

    #top
    print("TOP")
    top_dict = {}
    for wsc_id in top50idx:
        top_dict[wsc_id] = wsc_datapoints_new[wsc_id]
    for wsc_id, sentence in top_dict.items(): #[experiment][top50idx].to_dict().items():
        print("{}\t{}\t{}\t{}".format(wsc_id, wsc_index_prob_diffs[wsc_id], wsc_pair_ids.iloc[wsc_id], sentence))
        top_pair_ids.append(wsc_pair_ids.iloc[wsc_id])
    print('\n\n')

    #bottom
    print("BOTTOM")
    bottom_dict = {}
    for wsc_id in bottom50idx:
        bottom_dict[wsc_id] = wsc_datapoints_new[wsc_id]
    for wsc_id, sentence in bottom_dict.items(): #enumerate(wsc_datapoints_new[bottom50idx]): #[experiment][bottom50idx].to_dict().items():
        print("{}\t{}\t{}\t{}".format(wsc_id, wsc_index_prob_diffs[wsc_id], wsc_pair_ids.iloc[wsc_id], sentence))
        bottom_pair_ids.append(wsc_pair_ids.iloc[wsc_id])

    print('\n\n')

    print("CLOSEST")
    # smallest margin
    small_dict = {}
    for wsc_id in top50idx_abs:
        small_dict[wsc_id] = wsc_datapoints_new[wsc_id]
    for wsc_id, sentence in small_dict.items(): #enumerate(wsc_datapoints_new[top50idx_abs]): #[experiment][top50idx_abs].to_dict().items():
        print("{}\t{}\t{}\t{}".format(wsc_id, wsc_index_prob_diffs[wsc_id], wsc_pair_ids.iloc[wsc_id], sentence))
        small_pair_ids.append(wsc_pair_ids.iloc[wsc_id])
    print('\n\n')

    current_answers = answers[experiment]
    """

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
    """

    #set overlap
    top_pair_ids = set(top_pair_ids)
    bottom_pair_ids = set(bottom_pair_ids)
    small_pair_ids = set(small_pair_ids)
    t_b_intersection = top_pair_ids.intersection(bottom_pair_ids)

    #get pair acc.
    wsc_pair_ids_correct = defaultdict()
    for wsc_id, diff in enumerate(prob_diffs):
        pair_id = wsc_pair_ids.iloc[wsc_id]

        if pair_id not in wsc_pair_ids_correct.keys():
            if diff > 0:
                wsc_pair_ids_correct[pair_id] = 1
            else:
                wsc_pair_ids_correct[pair_id] = 0
        else:
            if diff > 0 and  wsc_pair_ids_correct[pair_id] == 1:
                wsc_pair_ids_correct[pair_id] = 1
            else:
                wsc_pair_ids_correct[pair_id] = 0


    count_pairs_correct = len([pair for pair in wsc_pair_ids_correct.values() if pair == 1])

    print("{} acc: {} pair acc: {} len_top_bottm_inter: {}  len_small_set: {}  ".
          format(experiment, accuracies[experiment]['all'] / len(current_indices), count_pairs_correct/ len(wsc_pair_ids), len(t_b_intersection), len(small_pair_ids)))


    all_accuracies.append(accuracies[experiment]['all'] / len(current_indices))

  