import torch
from pytorch_transformers import XLNetTokenizer, XLNetLMHeadModel
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
import re
import sys

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/new_test.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

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

# Load pre-trained model tokenizer (vocabulary)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
PADDING_TEXT = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(PADDING_TEXT)))

# perturbation: correct/wrong: original/altered
# this dupplicates original but whatever the fuck
description = {}
indices = {}
answers = {}

prediction_original = []
# Load pre-trained model (weights)
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
model.eval()

for current_alt, current_pron_index in [('text_original', 'pron_index'),
                                        ('text_voice', 'pron_index_voice'),
                                        ('text_context', 'pron_index_context'),
                                        ('text_adverb', 'pron_index_adverb'),
                                        ('text_tense', 'pron_index_tense'),
                                        ('text_number', 'pron_index_number'),
                                        ('text_gender', 'pron_index'),
                                        ('text_rel_1', 'pron_index_rel'),
                                        ('text_syn', 'pron_index_syn'),
                                        ('text_scrambled', 'pron_index_scrambled'),
                                        ('text_freqnoun', 'pron_index_freqnoun')]:
    description[current_alt] = {'correct': {'ans': [], 'dis': []}, 'wrong': {'ans': [], 'dis': []}}
    indices[current_alt] = {'ans': [], 'dis': []}
    answers[current_alt] = []

    correct_preds_enhanced, stability_match = 0, 0
    all_preds = 0

    print(current_alt)
    for q_index, dp_split in wsc_datapoints.iterrows():
        if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):
            # save the index
            # Tokenized input
            correct_answer = dp_split['correct_answer']
            text_enhanced = re.sub(r' +', ' ', dp_split[current_alt].lower())

            tokenized_enhanced_text = tokenizer.tokenize(text_enhanced)

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

            pronoun = pronoun.lower()

            discrim_word = dp_split['discrim_word']
            if isinstance(discrim_word, str):
                discrim_word = discrim_word.strip()
            else:
                discrim_word = None
                discrim_word_index = None

            pronoun_index_orig_enhanced =  int(dp_split[current_pron_index])
            tokenized_option_A = tokenizer.tokenize(tokens_pre_word_piece_A)
            tokenized_option_B = tokenizer.tokenize(tokens_pre_word_piece_B)
            tokenized_pronoun = tokenizer.tokenize(pronoun)

            tokenized_option_A_len = len(tokenized_option_A)
            tokenized_option_B_len = len(tokenized_option_B)

            ##print(tokenized_option_A, "tokenized_option A")
            ##print(tokenized_option_B, "tokenized_option B")

            if current_alt == 'text_number':
                tokenized_pronoun = tokenizer.tokenize(dp_split['pron_number'].strip().lower())
            elif current_alt == 'text_gender':
                tokenized_pronoun = tokenizer.tokenize(dp_split['pron_gender'].strip().lower())

            matched_pronouns_enhanced_text = find_sub_list(tokenized_pronoun,  tokenized_enhanced_text)
            first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])
            correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
            pronoun_index_text_enhanced = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]

            tokenized_text_enhanced_A = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_A)
            tokenized_text_enhanced_B = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_B)

            if discrim_word:
                tokenized_discrim_word = tokenizer.tokenize(discrim_word)
                discrim_word_index_enhanced_A = find_keyword(tokenized_discrim_word, tokenized_text_enhanced_A)
                discrim_word_index_enhanced_B = find_keyword(tokenized_discrim_word, tokenized_text_enhanced_B)
                if not (discrim_word_index_enhanced_A and discrim_word_index_enhanced_B):
                    discrim_word = None

            matched_A_text_enhanced = find_sub_list(tokenized_option_A, tokenized_text_enhanced_A)
            matched_B_text_enhanced = find_sub_list(tokenized_option_B, tokenized_text_enhanced_B)

            masked_indices_A_text_enhanced = [m for m in matched_A_text_enhanced if m[0] == pronoun_index_text_enhanced][0]
            masked_indices_B_text_enhanced = [m for m in matched_B_text_enhanced if m[0] == pronoun_index_text_enhanced][0]


            tokenized_text_A_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_A)
            tokenized_text_B_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_B)

            for masked_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
                tokenized_text_enhanced_A[masked_index] = '<mask>'

            for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                tokenized_text_enhanced_B[masked_index] = '<mask>'

            masked_lm_labels_A_enhanced = []
            masked_lm_labels_B_enhanced = []

            # process padding
            #enhanced
            indexed_tokens_A_enhanced = PADDING_TEXT + tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_A))
            indexed_tokens_B_enhanced = PADDING_TEXT + tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_B))
            indexed_tokens_A_pre_mask_enhanced = PADDING_TEXT + tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask_enhanced))
            indexed_tokens_B_pre_mask_enhanced = PADDING_TEXT + tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask_enhanced))

            # mask all labels but wsc options (enhanced)
            for token_index in range(len(indexed_tokens_A_enhanced)):
                if token_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
                    masked_lm_labels_A_enhanced.append(indexed_tokens_A_pre_mask_enhanced[token_index])
                else:
                    masked_lm_labels_A_enhanced.append(-1)

            # mask all labels but wsc options
            for token_index in range(len(indexed_tokens_B_enhanced)):
                if token_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                    masked_lm_labels_B_enhanced.append(indexed_tokens_B_pre_mask_enhanced[token_index])
                else:
                    masked_lm_labels_B_enhanced.append(-1)

            masked_tokens_A_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A_enhanced if i != -1]))
            masked_tokens_B_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B_enhanced if i != -1]))

            masked_lm_labels_A_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_A_enhanced) if item != -1]
            masked_lm_labels_B_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_B_enhanced) if item != -1]

            tokens_tensor_A_enhanced = torch.tensor([indexed_tokens_A_enhanced])
            masked_lm_labels_A_enhanced = torch.tensor([masked_lm_labels_A_enhanced])

            tokens_tensor_B_enhanced = torch.tensor([indexed_tokens_B_enhanced])
            masked_lm_labels_B_enhanced = torch.tensor([masked_lm_labels_B_enhanced])

            # If you have a GPU, put everything on cuda
            tokens_tensor_A_enhanced = tokens_tensor_A_enhanced.to(device=device)
            tokens_tensor_B_enhanced = tokens_tensor_B_enhanced.to(device=device)
            masked_lm_labels_A_enhanced = masked_lm_labels_A_enhanced.to(device=device)
            masked_lm_labels_B_enhanced = masked_lm_labels_B_enhanced.to(device=device)

            model.to(device=device)

            total_logprobs_A_enhanced = 0
            total_logprobs_B_enhanced = 0

            def get_masks(input_tensor, mask_tuple):
                num_tokens = input_tensor.size(1)
                to_mask = [i[0] for i in mask_tuple]
                num_tokens_to_predict = len(to_mask)

                perm_mask = torch.zeros(1, num_tokens, num_tokens)
                perm_mask[:, :, to_mask] = 1.0

                target_mapping = torch.zeros(1, num_tokens_to_predict, num_tokens)
                in_out_map = zip(range(num_tokens_to_predict), to_mask)
                for i, j in in_out_map:
                    target_mapping[0, i, j] = 1.0

                return perm_mask.to(device=device), target_mapping.to(device=device)

            with torch.no_grad():
                perm_mask_A, target_mapping_A = get_masks(tokens_tensor_A_enhanced, masked_lm_labels_A_non_neg_enhanced)
                perm_mask_B, target_mapping_B = get_masks(tokens_tensor_B_enhanced, masked_lm_labels_B_non_neg_enhanced)
                probs_A_enhanced = model(tokens_tensor_A_enhanced, perm_mask=perm_mask_A, target_mapping=target_mapping_A)
                probs_B_enhanced = model(tokens_tensor_B_enhanced, perm_mask=perm_mask_B, target_mapping=target_mapping_B)
                probs_A_enhanced = probs_A_enhanced[0].to(device=device)
                probs_B_enhanced = probs_B_enhanced[0].to(device=device)

                logprobs_A_enhanced = torch.nn.functional.log_softmax(probs_A_enhanced, dim=-1)
                logprobs_B_enhanced = torch.nn.functional.log_softmax(probs_B_enhanced, dim=-1)

                probs_array_A_enhanced = []
                probs_array_B_enhanced = []

                # A
                for n, index_item in enumerate(masked_lm_labels_A_non_neg_enhanced):
                    index, item = index_item
                    total_logprobs_A_enhanced += logprobs_A_enhanced[0, n, item].item()
                    #probs_array_A_enhanced = logprobs_A_enhanced[0, torch.arange(len(indexed_tokens_A_enhanced)), indexed_tokens_A_enhanced]

                # B
                for n, index_item in enumerate(masked_lm_labels_B_non_neg_enhanced):
                    index, item = index_item
                    total_logprobs_B_enhanced += logprobs_B_enhanced[0, n, item].item()
                    #probs_array_B_enhanced = logprobs_B_enhanced[0, torch.arange(len(indexed_tokens_B_enhanced)), indexed_tokens_B_enhanced]

                # prob shift
                c = total_logprobs_A_enhanced / tokenized_option_A_len
                w = total_logprobs_B_enhanced / tokenized_option_B_len

                if correct_answer == 'B':
                    c, w = w, c

                description[current_alt]['correct']['ans'].append(c)
                description[current_alt]['wrong']['ans'].append(w)
                indices[current_alt]['ans'].append(q_index)

                max_index_enhanced = np.argmax([total_logprobs_A_enhanced / tokenized_option_A_len, total_logprobs_B_enhanced
                                                / tokenized_option_B_len ])

                prediction_enhanced = "A" if max_index_enhanced == 0 else "B"

                if prediction_enhanced == correct_answer.strip().strip('.').replace(' ', ''):
                    answers[current_alt].append(1)
                    correct_preds_enhanced += 1
                else:
                    answers[current_alt].append(0)

                # TODO
                if current_alt == 'text_original':
                    prediction_original.append(prediction_enhanced)

                else:
                    if prediction_enhanced == prediction_original[q_index]:
                        stability_match += 1

                all_preds += 1
                #print("#############################################################################")
        else:
            if current_alt == 'text_original':
                print("broken code m8")
                exit()

            continue

    accuracy_enhanced = correct_preds_enhanced/all_preds
    print("accuracy: {}/{} = {}".format(correct_preds_enhanced, all_preds, accuracy_enhanced))
    print("stability: {}/{} = {}%".format(stability_match, all_preds, stability_match / all_preds))

    description[current_alt]['accuracy'] = accuracy_enhanced
    description[current_alt]['stability'] = stability_match / all_preds

#print(description)
with open('description_dump.pickle', 'wb') as f:
    pickle.dump((description, indices, answers), f)
