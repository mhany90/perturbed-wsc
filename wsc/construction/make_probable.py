import torch
import numpy as np
from copy import deepcopy
from random import shuffle, sample
import math
import pandas as pd
import re

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')


path_probale_nouns = '../../data/resources/nounlist.txt'
nouns_file = open(path_probale_nouns , 'r')
nouns_list = nouns_file.readlines()


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results

def replace_pronoun(tokenized_text, pronoun_index, tokenized_option):
    tokenized_text = tokenized_text[:pronoun_index] + tokenized_option + tokenized_text[pronoun_index:]
    new_pronoun_index = pronoun_index + len(tokenized_option)
    tokenized_text.pop(new_pronoun_index)
    return tokenized_text

def insert_probable(text, proportion, pronoun_index):
    text_tokenized = text.split()
    text_with_word = deepcopy(text_tokenized)
    pronoun_index_with_word = pronoun_index

    text_no_words = len(text_tokenized)
    num_insertions = math.ceil(text_no_words * proportion)

    #to scramble or not
    for _ in range(num_insertions):
        #sample index
        insert_index = sample(range(text_no_words), 1)[0]
        sampled_word = sample(nouns_list, 1)[0].replace('\n', '')
        if insert_index <= pronoun_index:
            pronoun_index_with_word = pronoun_index_with_word + 1
        text_with_word.insert(insert_index, sampled_word)
    return ' '.join(text_with_word), pronoun_index_with_word


for q_index, dp_split in wsc_datapoints.iterrows():
        # Tokenized input

        pronoun = dp_split['pron'].strip()
        pronoun_index_orig = dp_split['pron_index']

        #check for empty
        text = dp_split['text_original'].strip().replace('\n', '')
        # match pron
        matched_pronouns = find_sub_list([pronoun], text.split())

        first_indices = np.array([mp[0] for mp in matched_pronouns])
        correct_idx = (np.abs(first_indices - pronoun_index_orig)).argmin()
        pronoun_index = matched_pronouns[correct_idx][0]
        print(pronoun_index, "pronoun_index")

        text_with_word, pronoun_index_with_word = insert_probable(text, 0.1, pronoun_index)
        print(pronoun_index_with_word, "pronoun_index_with_word")

        wsc_datapoints.loc[q_index, 'text_freqnoun'] = text_with_word
        wsc_datapoints.loc[q_index, 'pron_index_freqnoun'] = pronoun_index_with_word

wsc_datapoints.to_csv(path_or_buf='../../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.tsv',sep='\t')
