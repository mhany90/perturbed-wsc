from nltk.corpus import wordnet as wn
from nltk import pos_tag
import numpy as np


NOUNS = ['NN', 'NNP', 'NNS', 'NNPS']
path_to_wsc = '../data/wsc_data/enhanced.dset.tsv'
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


print(wsc_datapoints[0])
for dp in wsc_datapoints[1:]:
    dp_split = dp.split('\t')
    if dp_split[1].replace(' ', '') != '-':
        text = dp_split[0]
        tokenized_pronoun = dp_split[4]
        pronoun_index_orig =  int(dp_split[5].strip())

        tokenized_pronoun = tokenized_pronoun.lower().strip()

        text = text.lower()

        tokens_A_orig = dp_split[8]
        tokens_A_orig = tokens_A_orig.lower().strip()

        tokens_B_orig = dp_split[10].lower()
        tokens_B_orig = tokens_B_orig.lower().strip()


        tokens_A_syn = dp_split[9].lower()
        tokens_A_syn = tokens_A_syn.lower().strip()

        tokens_B_syn = dp_split[11].lower()
        tokens_B_syn = tokens_B_syn.lower().strip()



        #print( tokens_A_orig , " tokens_A_orig ")
        #print( tokens_A_syn , " tokens_A_syn ")


        text_new_A = text.replace(tokens_A_orig, tokens_A_syn)
        text_new_B = text_new_A.replace(tokens_B_orig, tokens_B_syn)
        #print(text_new_B, "text_new_B")

        text_new_B_tok = text_new_B.split()
        #print(tokenized_pronoun, text_new_B_tok, "tokenized_pronoun, text_new_B_tok")
        matched_pronouns_text = find_sub_list([tokenized_pronoun], text_new_B_tok)
        #print(matched_pronouns_text, ' matched_pronouns_text')
        first_indices_text = np.array([mp[0] for mp in matched_pronouns_text])

        #print(first_indices_text, pronoun_index_orig, " first_indices_text, pronoun_index_orig ")
        correct_idx_text = (np.abs(first_indices_text - pronoun_index_orig)).argmin()
        pronoun_index_text = matched_pronouns_text[correct_idx_text][0]


        #print( pronoun_index_text, " pronoun_index_text")


        dp_split[7] = str(pronoun_index_text)
        dp_split[2] = text_new_B

        print('\t'.join(dp_split), end = '')


