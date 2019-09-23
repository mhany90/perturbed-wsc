import torch
import numpy as np
from numpy.random import binomial
from random import shuffle
import pandas as pd


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../../data/wsc_data/enhanced.tense.random.role.syn.voice.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')
#wsc_file = open(path_to_wsc, 'r')
#wsc_datapoints = wsc_file.readlines()

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

def scramble(text, proportion, pron):
    text_tokenized = text.split()
    text_scrambled = []
    text_no_words = len(text_tokenized)
    #to scramble or not
    for index, word in enumerate(text_tokenized):
        bvar = binomial(1, proportion)
        if bvar < 0.1 or word == pron:
            text_scrambled.append(word)
        else:
            wordList = list(word)
            wordList_copy = wordList[1:-1]
            shuffle(wordList_copy)
            #partial shuffle
            wordList[1:-1] = wordList_copy
            text_scrambled.append(''.join(wordList))
    return ' '.join(text_scrambled)


for q_index, dp_split in wsc_datapoints.iterrows():
        # Tokenized input

        correct_answer = dp_split['correct_answer'].strip()
        pronoun = dp_split['pron'].strip()
        pronoun_index_orig = dp_split['pron_index']
        pronoun_index_orig_enhanced = pronoun_index_orig

        #check for empty
        text = dp_split['text_original'].strip()

        #match pron
        matched_pronouns_scramble = find_sub_list([pronoun], text.split())
        first_indices_scramble = np.array([mp[0] for mp in  matched_pronouns_scramble])
        correct_idx_scramble = (np.abs(first_indices_scramble - pronoun_index_orig)).argmin()

        text_enhanced = scramble(dp_split[0], 0.5, pronoun)
        print(text, "text")
        print(text_enhanced, ' text_enhanced')

        matched_pronouns_enhanced_text = find_sub_list([pronoun],  text.split())
        first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])
        print(matched_pronouns_enhanced_text, "matched_pronouns_text_enhanced")
        correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
        print(correct_idx_text_enhanced, " correct_idx_text_enhanced")
        pronoun_index_text_enhanced  = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]

        wsc_datapoints.loc[q_index, 'text_scrambled'] = text_enhanced
        wsc_datapoints.loc[q_index, 'pron_index_scrambled'] = pronoun_index_text_enhanced

wsc_datapoints.to_csv(path_or_buf='../../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.tsv',sep='\t')

