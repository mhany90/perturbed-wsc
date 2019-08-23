from nltk.corpus import wordnet as wn
from nltk import pos_tag
import numpy as np
import random


path_to_wsc = '../data/wsc_data/enhanced.dset.tsv'
path_to_wsc_enh = '../data/wsc_data/enhanced.voice.syn.rel.wsc.tsv'

wsc_file = open(path_to_wsc, 'r')
wsc_datapoints = wsc_file.readlines()

wsc_file_enh = open(path_to_wsc_enh, 'r')
wsc_datapoints_enh = wsc_file_enh.readlines()


#rel clause templates
templates = ["who we had discussed", "who he had discussed", "who she had discussed", "who you had discussed",
 "which we had seen", "which he had seen", "which she had seen", "which you had seen",
 "who we know from", "who he knows from", "who she knows from", "who you know from",
 "that is mentioned in", "that is located at", "that is close to", "that is known for"]

def replace_pronoun(tokenized_text, pronoun_index, tokenized_option):
    tokenized_text = tokenized_text[:pronoun_index] + tokenized_option + tokenized_text[pronoun_index:]
    new_pronoun_index = pronoun_index + len(tokenized_option)
    tokenized_text.pop(new_pronoun_index)
    return tokenized_text

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results

all_datapoints = []
all_enh_texts = [dp.split('\t')[0].strip() for dp in wsc_datapoints_enh]

for index, dp in enumerate(wsc_datapoints[1:]):
    dp_split = dp.split('\t')
    text = dp_split[0]
    if text.strip() in all_enh_texts:
        correct_enh_index = all_enh_texts.index(text)
        all_datapoints.append(wsc_datapoints_enh[correct_enh_index])
    else:
        all_datapoints.append(dp)


file_out = open('enhanced.new.wsc.restored.tsv', 'w')
file_out.write(wsc_datapoints[0])
for dp in all_datapoints:
    file_out.write(dp)

file_out.close()