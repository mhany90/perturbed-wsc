from nltk.corpus import brown
from nltk import pos_tag
import numpy as np
import random
from nltk.corpus import brown


path_to_wsc = 'enhanced.new.wsc.random.twosents.tsv'
wsc_file = open(path_to_wsc, 'r')
wsc_datapoints = wsc_file.readlines()

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

sents = brown.sents(categories=['news', 'editorial', 'reviews'])

for dp in wsc_datapoints[1:]:
    dp_split = dp.split('\t')
    text = dp_split[0]
    text_tokenized = text.split()

    pronoun_index_orig =  int(dp_split[7].strip())

    chosen_sent_1 = random.choice(sents)
    chosen_sent_2 = random.choice(sents)
    chosen_sent_3 = random.choice(sents)
    chosen_sent_4 = random.choice(sents)

    len_addition = len(chosen_sent_1) + len(chosen_sent_2)
    pronoun_index_text = pronoun_index_orig + len_addition

    text_with_random_sents = chosen_sent_1 + chosen_sent_2 + text_tokenized + chosen_sent_3 + chosen_sent_4

    #print(text_with_random_sents)

    dp_split[11] = str(pronoun_index_text)
    dp_split[1] = " ".join(text_with_random_sents)

    all_datapoints.append('\t'.join(dp_split))
    print('\t'.join(dp_split), end = '')



file_out = open('enhanced.new.wsc.random.twosents.tsv', 'w')
file_out.write(wsc_datapoints[0])
for dp in all_datapoints:
    file_out.write(dp)

file_out.close()