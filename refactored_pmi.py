import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
import sys
from scipy import spatial
from scipy.sparse import dok_matrix, csr_matrix



# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = 'data/final.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

path_to_pmi = '../../perturbs/pmi_relevant.pickle'
path_to_pmi_contexts_vocab = '../../perturbs/pmi.contexts.vocab'
pmi_dict = pickle.load(open(path_to_pmi, 'rb'))

pmi_contexts_vocab = open(path_to_pmi_contexts_vocab, 'r').readlines()
pmi_contexts_vocab = [p.strip().strip('\n').replace(' ', '').lower() for p in pmi_contexts_vocab]

def find_keyword(tokens, text):
    result = []
    for i in tokens:
        try:
            result.append(text.index(i))
        except:
            return None

    # return np.array(result)
    # get just the last token for now
    return result[-1]

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

def get_pmi(pmi_dict, pmi_contexts_vocab, tokenized_text1, tokenized_text2, apply_log=True):
    total_pmi = 0
    count_oov = 0
    for word1 in tokenized_text1:
        if word1 in pmi_dict.keys():
           for word2 in tokenized_text2:
               if word2 in pmi_contexts_vocab:
                    idx_word2 = pmi_contexts_vocab.index(word2)
                    if apply_log:
                        pmi_w1_w2 = np.log(pmi_dict[word1][0][idx_word2])
                        #np.where(pmi_dict[word1][0][idx_word2] != 0, np.log(pmi_dict[word1][0][idx_word2]), 0)
                        #pmi_w1_w2[pmi_w1_w2 == -np.inf] = 0.0
                        #pmi_w1_w2[pmi_w1_w2 == np.inf] = 0.0
                    else:
                        pmi_w1_w2 = pmi_dict[word1][0][idx_word2]
                    total_pmi += pmi_w1_w2
               else:
                    count_oov += 1
        else:
            count_oov += 1


    return total_pmi, count_oov

def get_pmi_vec(pmi_dict, pmi_contexts_vocab, tokenized_text1, tokenized_text2, apply_log=True):
    total_sim = 0
    for word1 in tokenized_text1:
        if word1 in pmi_dict.keys():
           for word2 in tokenized_text2:
               if word2 in pmi_dict.keys():
                    if apply_log:
                        pmi_vec1 = np.ma.log(pmi_dict[word1][0])
                        pmi_vec2 = np.ma.log(pmi_dict[word2][0])
                    else:
                        pmi_vec1 = pmi_dict[word1][0]
                        pmi_vec2 = pmi_dict[word2][0]
                    sim = np.dot(pmi_vec1, pmi_vec2)
                    total_sim += sim
    return total_sim

def get_ppmi_vec(pmi_dict, pmi_contexts_vocab, tokenized_text1, tokenized_text2, apply_log=False, neg=True):
    total_sim = 0
    for word1 in tokenized_text1:
        if word1 in pmi_dict.keys():
           for word2 in tokenized_text2:
               if word2 in pmi_dict.keys():
                    if apply_log:
                        pmi_vec1 = np.log(pmi_dict[word1][0])
                        pmi_vec1[pmi_vec1 < 0] = 0
                        pmi_vec2 = np.log(pmi_dict[word2][0])
                        pmi_vec2[pmi_vec2 < 0] = 0
                    else:
                        pmi_vec1 = pmi_dict[word1][0]
                        if neg:
                            pmi_vec1 -= np.log(1)
                        pmi_vec1[pmi_vec1 < 0] = 0
                        pmi_vec2 = pmi_dict[word2][0]
                        if neg:
                            pmi_vec2 -= np.log(1)
                        pmi_vec2[pmi_vec2 < 0] = 0

                    sim = np.dot(pmi_vec1, pmi_vec2.T)
                    total_sim += sim
    return total_sim

# perturbation: correct/wrong: original/altered
# this dupplicates original but whatever the fuck
description = {}
indices = {}
answers = {}
attentions = {}
prediction_original = []
baseline_attentions = []
all_pmi_diffs = {}
accuracies, stabilities, counts = {}, {}, {}

for current_alt, current_pron_index in [('text_original', 'pron_index'),
                                        ('text_voice', 'pron_index_voice'),
                                        ('text_tense', 'pron_index_tense'),
                                        ('text_context', 'pron_index_context'),
                                        ('text_number', 'pron_index_number'),
                                        ('text_gender', 'pron_index'),
                                        ('text_rel_1', 'pron_index_rel'),
                                        ('text_syn', 'pron_index_syn'),
                                        ('text_scrambled', 'pron_index_scrambled'),
                                        ('text_freqnoun', 'pron_index_freqnoun'),
                                        ('text_adverb', 'pron_index_adverb')
                                        ]:

    accuracies[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}
    stabilities[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}
    counts[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}
    description[current_alt] = {'correct': {'ans': [], 'dis': [], 'attn': []}, 'wrong': {'ans': [], 'dis': [], 'attn': []},
                                'all':{'pron_attn': []}}
    indices[current_alt] = {'ans': [], 'dis': []}
    answers[current_alt] = []
    all_pmi_diffs[current_alt] = []


    correct_preds_enhanced, stability_match, count_oov, total_pmi_diff = 0, 0, 0, 0.0
    all_preds = 0

    print(current_alt)
    for q_index, dp_split in wsc_datapoints.iterrows():
        if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):
            # save the index
            # Tokenized input
            correct_answer = dp_split['correct_answer'].strip().strip('.').replace(' ', '')
            text_enhanced = dp_split[current_alt]
            tokenized_enhanced_text = [t.strip().strip('\n').replace(' ', '').lower() for t in text_enhanced.split()]
            if current_alt == 'text_syn':
                tokens_pre_word_piece_A = dp_split['answer_a_syn']
                tokens_pre_word_piece_B = dp_split['answer_b_syn']

            elif current_alt == 'text_gender':
                tokens_pre_word_piece_A = dp_split['answer_a_gender']
                tokens_pre_word_piece_B = dp_split['answer_b_gender']

            elif current_alt == 'text_number':
                tokens_pre_word_piece_A = dp_split['answer_a_number']
                tokens_pre_word_piece_B = dp_split['answer_b_number']

            else:
                tokens_pre_word_piece_A = dp_split['answer_a']
                tokens_pre_word_piece_B = dp_split['answer_b']

            if current_alt == 'text_gender':
                pronoun = dp_split['pron_gender'].strip()
            elif current_alt == 'text_number':
                pronoun = dp_split['pron_number'].strip()
            else:
                pronoun = dp_split['pron'].strip()

            discrim_word = dp_split['discrim_word']
            if isinstance(discrim_word, str):
                discrim_word = discrim_word.strip()
            else:
                discrim_word = None
                discrim_word_index = None

            pronoun_index_orig_enhanced =  int(dp_split[current_pron_index])
            tokenized_option_A = [t.strip().strip('\n').replace(' ', '').lower() for t in tokens_pre_word_piece_A.split()]
            tokenized_option_B = [t.strip().strip('\n').replace(' ', '').lower() for t in tokens_pre_word_piece_B.split()]
            tokenized_pronoun = pronoun.strip().lower()

            #print(tokenized_option_A , " tokenized_option_A ")
            #print(tokenized_option_B , " tokenized_option_B ")

            tokenized_option_A_len = len(tokenized_option_A)
            tokenized_option_B_len = len(tokenized_option_B)

            if current_alt == 'text_number':
                tokenized_pronoun = dp_split['pron_number'].strip()
            elif current_alt == 'text_gender':
                tokenized_pronoun = dp_split['pron_gender'].strip()

            #matched_pronouns_enhanced_text = find_sub_list(tokenized_pronoun,  tokenized_enhanced_text)
            #first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])
            #correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
            #pronoun_index_text_enhanced = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]

            #tokenized_text_enhanced_A = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_A)
            #tokenized_text_enhanced_B = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_B)


            tokenized_discrim_word = [t.strip().strip('\n').replace(' ', '').strip().lower() for t in discrim_word.split()]

            pmi_A, _ = get_pmi(pmi_dict, pmi_contexts_vocab,  tokenized_option_A,  tokenized_discrim_word)
            pmi_B, _ = get_pmi(pmi_dict, pmi_contexts_vocab,  tokenized_option_B,  tokenized_discrim_word)

            #pmi_A = get_ppmi_vec(pmi_dict, pmi_contexts_vocab, tokenized_option_A,  tokenized_enhanced_text)
            #pmi_B = get_ppmi_vec(pmi_dict, pmi_contexts_vocab, tokenized_option_B,  tokenized_enhanced_text)

            #count_oov += count_oov_this
            c = pmi_A
            w = pmi_B

            if correct_answer == 'B':
                c, w = w, c

            description[current_alt]['correct']['ans'].append(c)
            description[current_alt]['wrong']['ans'].append(w)
            indices[current_alt]['ans'].append(q_index)

            """
            matched_A_text_enhanced = find_sub_list(tokenized_option_A, tokenized_text_enhanced_A)
            matched_B_text_enhanced = find_sub_list(tokenized_option_B, tokenized_text_enhanced_B)

            masked_indices_A_text_enhanced = [m for m in matched_A_text_enhanced if m[0] == pronoun_index_text_enhanced][0]
            masked_indices_B_text_enhanced = [m for m in matched_B_text_enhanced if m[0] == pronoun_index_text_enhanced][0]

            tokenized_text_A_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_A)
            tokenized_text_B_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_B)
            """

            len_tokens_A = len(tokenized_option_A)
            len_tokens_B = len(tokenized_option_B)

            correct_answer = correct_answer.strip().strip('.').replace(' ', '')

            if correct_answer == 'A':
                pmi_diff = pmi_A / float(len_tokens_A) - pmi_B  /  float(len_tokens_B)
            else:
                pmi_diff = pmi_B /  float(len_tokens_B) - pmi_A /  float(len_tokens_A)

            all_pmi_diffs[current_alt].append(pmi_diff)
            total_pmi_diff += pmi_diff



            prediction_enhanced = "A" if  pmi_A / len_tokens_A > pmi_B / len_tokens_B else "B"

            if prediction_enhanced == correct_answer:
                answers[current_alt].append(1)
                accuracies[current_alt]['all'] += 1

                if dp_split['associative'] == 1:
                    accuracies[current_alt]['associative'] += 1
                else:
                    accuracies[current_alt]['!associative'] += 1

                if dp_split['switchable'] == 1:
                    accuracies[current_alt]['switchable'] += 1
                else:
                    accuracies[current_alt]['!switchable'] += 1

            else:
                answers[current_alt].append(0)

                # TODO
            if current_alt == 'text_original':
                prediction_original.append(prediction_enhanced)
            else:
                if prediction_enhanced == prediction_original[q_index]:
                    stabilities[current_alt]['all'] += 1

                    if dp_split['associative'] == 1:
                        stabilities[current_alt]['associative'] += 1
                    else:
                        stabilities[current_alt]['!associative'] += 1

                    if dp_split['switchable'] == 1:
                        stabilities[current_alt]['switchable'] += 1
                    else:
                        stabilities[current_alt]['!switchable'] += 1

            all_preds += 1

            if dp_split['associative'] == 1:
                counts[current_alt]['associative'] += 1
            else:
                counts[current_alt]['!associative'] += 1

            if dp_split['switchable'] == 1:
                counts[current_alt]['switchable'] += 1
            else:
                counts[current_alt]['!switchable'] += 1

        else:
            if current_alt == 'text_original':
                print("broken code m8")
                exit()

            continue

    # add the total just in case
    counts[current_alt]['all'] = all_preds

    accuracy_enhanced = correct_preds_enhanced/all_preds
    print("accuracy: {}/{} = {}".format(accuracies[current_alt]['all'], all_preds, accuracy_enhanced))
    print("stability: {}/{} = {}%".format(stabilities[current_alt]['all'], all_preds, stability_match / all_preds))
    print(count_oov, ' : count_oov')
    print(total_pmi_diff / all_preds, ' : total pmi diff')

#print(description)
with open('description_dump_pmi_log.pickle', 'wb') as f:
    pickle.dump((description, indices, answers, counts, accuracies, stabilities, all_pmi_diffs), f)
