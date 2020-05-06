import pandas as pd
import tqdm
import numpy as np
from helpers import find_sublist
import json
import os

TSV_PATH = '../data/final.tsv'
OUT_PATH = '../data/winogrande'
EXPERIMENT_ARR = [('text_original', 'pron_index'),
                  ('text_voice', 'pron_index_voice'),
                  ('text_tense', 'pron_index_tense'),
                  ('text_number', 'pron_index_number'),
                  ('text_gender', 'pron_index'),
                  ('text_rel_1', 'pron_index_rel'),
                  ('text_syn', 'pron_index_syn'),
                  ('text_adverb', 'pron_index_adverb')]

datafile = pd.read_csv(TSV_PATH, sep='\t')

for m, (exp_name, pron_col) in enumerate(EXPERIMENT_ARR):
    dumpfile = open(os.path.join(OUT_PATH, exp_name + ".jsonl"), 'w')
    for n, (q_index, entry) in enumerate(datafile.iterrows()):
        current_dict = {}
        if entry[exp_name].replace(' ', '') in [None, '-']:
            continue

        text = entry[exp_name].split(" ")
        text_uncased = [i.lower() for i in text]

        suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_gender', 'text_number'] else ""
        pronoun = [entry['pron{}'.format(suffix)].lower()]

        pron_index = int(entry[pron_col])
        matched_prons = find_sublist(pronoun, text_uncased)
        best_match = (np.abs(matched_prons - pron_index)).argmin()
        pron_index = matched_prons[best_match]

        text[pron_index] = '_'
        text = " ".join(text).lower()

        suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_syn', 'text_gender', 'text_number'] else ""
        text_option_A = entry['answer_a{}'.format(suffix)]
        text_option_B = entry['answer_b{}'.format(suffix)]

        correct_answer = entry['correct_answer'].strip().strip('.').replace(' ', '')
        correct_answer = str(["a", "b"].index(correct_answer.lower()) + 1)

        current_dict['sentence'] = text
        current_dict['option1'] = text_option_A
        current_dict['option2'] = text_option_B
        current_dict['answer'] = correct_answer
        current_dict['pair_number'] = entry['pair_number']
        current_dict['qID'] = "q{}{}".format(m, n)
        try:
            current_dict['associative'] = int(entry['associative'])
            current_dict['switchable'] = int(entry['switchable'])
        except:
            current_dict['associative'] = 0
            current_dict['switchable'] = 0

        dumpfile.write(json.dumps(current_dict) + "\n")

    dumpfile.close()
