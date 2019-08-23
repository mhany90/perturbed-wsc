from nltk.corpus import wordnet as wn
from nltk import pos_tag
import numpy as np
import random


path_to_wsc = '../data/wsc_data/enhanced.dset.tsv'
wsc_file = open(path_to_wsc, 'r')
wsc_datapoints = wsc_file.readlines()

#rel clause templates
templates = ["who we had discussed", "who he had discussed", "who she had discussed", "who you had discussed",
 "which we had seen", "which he had seen", "which she had seen", "which you had seen",
 "who we know from", "who he knows from", "who she knows from", "who you know from",
 "that is mentioned in", "that is located at", "that is close to", "that is known for", "which had been",
"who you met", "that is", "which was put there"]

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

for dp in wsc_datapoints[1:]:
    dp_split = dp.split('\t')
    if dp_split[1].replace(' ', '') != '-':
        text = dp_split[0]
        tokenized_pronoun = dp_split[4]
        pronoun_index_orig =  int(dp_split[5].strip())
        tokenized_pronoun = tokenized_pronoun.lower().strip()

        text_tokenized = text.lower().split()

        tokens_A_orig = dp_split[9]
        tokens_A_orig_tokenized = tokens_A_orig.lower().strip().split()

        tokens_B_orig = dp_split[11].lower()
        tokens_B_orig_tokenized = tokens_B_orig.lower().strip()

        try:
            #match option
            matched_option_A_text = find_sub_list(tokens_A_orig_tokenized, text_tokenized)
            first_indices_option_A = np.array([mp[0] for mp in matched_option_A_text])
            correct_idx_option_A = first_indices_option_A[0] + len(tokens_A_orig_tokenized)

            text_tokenized.insert(correct_idx_option_A, ', $$$ ,')
            print(text_tokenized, "text_tokenized $$$")
            while True:
                rel_clause_template = random.choice(templates)
                rel_clause_extension = str(input(rel_clause_template))
                if rel_clause_extension != 'x':
                    break

            rel_clause = rel_clause_template + rel_clause_extension
            text_rejoined = ' '.join(text_tokenized)
            text_with_clause = text_rejoined.replace('$$$', rel_clause)

            matched_pronouns_text = find_sub_list([tokenized_pronoun], text_with_clause.split())
            first_indices_text = np.array([mp[0] for mp in matched_pronouns_text])

            correct_idx_text = (np.abs(first_indices_text - pronoun_index_orig)).argmin()
            pronoun_index_text = matched_pronouns_text[correct_idx_text][0]

            dp_split[8] = str(pronoun_index_text)
            dp_split[1] = text_with_clause

            all_datapoints.append('\t'.join(dp_split))
            print('\t'.join(dp_split), end = '')
        except:
            continue



file_out = open('enhanced.new.wsc.tsv', 'w')
file_out.write(wsc_datapoints[0])
for dp in all_datapoints:
    file_out.write(dp)

file_out.close()