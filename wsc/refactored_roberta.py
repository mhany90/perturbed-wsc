import torch
from pytorch_transformers import RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import numpy as np
from copy import deepcopy
import re
import pandas as pd
import pickle


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')


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
    sl = [item for item in sl]
    l = [item for item in l]
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
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# perturbation: correct/wrong: original/altered
# this dupplicates original but whatever the fuck
description = {}
indices = {}
answers = {}
prediction_original = []


# Load pre-trained model (weights)
model = RobertaForMaskedLM.from_pretrained('roberta-large')
model.eval()

accuracies = {}
stabilities = {}
counts = {}

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
    accuracies[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}
    stabilities[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}
    counts[current_alt] = {'all': 0, 'switchable': 0, 'associative': 0, '!switchable': 0, '!associative': 0}

    correct_preds_enhanced = 0
    stability_match = 0
    all_preds = 0
    print(current_alt)


    for q_index, dp_split in wsc_datapoints.iterrows():
        if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):
            # save the index
            # Tokenized input
            correct_answer = dp_split['correct_answer']
            text_enhanced = dp_split[current_alt].lower()
            text_enhanced = re.sub(' +', ' ', text_enhanced)

            tokenized_enhanced_text = tokenizer.encode(text_enhanced, add_special_tokens=True)

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
                pronoun = 'because ' + dp_split['pron_gender'].strip()
            elif current_alt == 'text_number':
                pronoun = 'because ' + dp_split['pron_number'].strip()
            else:
                pronoun = 'because ' + dp_split['pron'].strip()

            pronoun = pronoun.lower()

            tokens_pre_word_piece_A = dp_split['answer_a'].strip().lower()
            tokens_pre_word_piece_B = dp_split['answer_b'].strip().lower()
            #print(tokens_pre_word_piece_A , " tokens_pre_word_piece_A ")
            #print(tokens_pre_word_piece_B , " tokens_pre_word_piece_B ")

            discrim_word = dp_split['discrim_word']
            if isinstance(discrim_word, str):
                discrim_word = discrim_word.strip()
            else:
                discrim_word = None
                discrim_word_index = None


            pronoun_index_orig_enhanced =  int(dp_split[current_pron_index])

            tokenized_option_A = tokenizer.encode(tokens_pre_word_piece_A, add_special_tokens=True)[1:-1]
            tokenized_option_B = tokenizer.encode(tokens_pre_word_piece_B, add_special_tokens=True)[1:-1]
            tokenized_pronoun = tokenizer.encode(pronoun, add_special_tokens=True)
            #print(tokenized_pronoun, "tokenized_pronoun")

            tokenized_option_A_len = len(tokenized_option_A)
            tokenized_option_B_len = len(tokenized_option_B)

            #print(tokenized_option_A, "tokenized_option A")
            #print(tokenized_option_B, "tokenized_option B")

            matched_pronouns_enhanced_text = find_sub_list([tokenized_pronoun[-2]],  tokenized_enhanced_text)
            first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])
            #print(matched_pronouns_enhanced_text, "matched_pronouns_text_enhanced")
            correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
            #print(correct_idx_text_enhanced, " correct_idx_text_enhanced")
            pronoun_index_text_enhanced  = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]
            #print(pronoun_index_text_enhanced, " pronoun_index_text_enhanced")

            tokenized_text_enhanced_A = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_A)
            tokenized_text_enhanced_B = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_B)

            if discrim_word:
                tokenized_discrim_word = tokenizer.tokenize(discrim_word)
                discrim_word_index_enhanced_A = find_keyword(tokenized_discrim_word, tokenized_text_enhanced_A)
                discrim_word_index_enhanced_B = find_keyword(tokenized_discrim_word, tokenized_text_enhanced_B)
                if not (discrim_word_index_enhanced_A and discrim_word_index_enhanced_B):
                    discrim_word = None

            #print(tokenized_text_enhanced_A, "tokenized_text_enhanced_A")

            matched_A_text_enhanced = find_sub_list(tokenized_option_A, tokenized_text_enhanced_A)
            matched_B_text_enhanced = find_sub_list(tokenized_option_B, tokenized_text_enhanced_B)

            #print(matched_A_text_enhanced, "matched A enhanced")

            masked_indices_A_text_enhanced = [m for m in matched_A_text_enhanced if m[0] == pronoun_index_text_enhanced][0]
            masked_indices_B_text_enhanced = [m for m in matched_B_text_enhanced if m[0] == pronoun_index_text_enhanced][0]

            #get index item

            masked_indices_items_A_text_enhanced = [(index, item) for index, item in
                                           zip(range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1] + 1),tokenized_option_A)]
            masked_indices_items_B_text_enhanced = [(index, item) for index, item in
                                                    zip(range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1] + 1),
                                                        tokenized_option_B)]



            for masked_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
                tokenized_text_enhanced_A[masked_index] = tokenizer.encode('<mask>')[0]
            #print(tokenized_text_enhanced_A, "tokenized_enchanced_text A MASKED")

            for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                tokenized_text_enhanced_B[masked_index] = tokenizer.encode('<mask>')[0]
            #print(tokenized_text_enhanced_B, "tokenized_enchanced_text B MASKED")



            #enhanced
            indexed_tokens_A_enhanced = tokenized_text_enhanced_A #tokenizer.encode(' '.join(tokenized_text_enhanced_A), add_special_tokens=True)
            indexed_tokens_B_enhanced = tokenized_text_enhanced_B #tokenizer.encode(' '.join(tokenized_text_enhanced_B), add_special_tokens=True)

            #enhanced
            tokens_tensor_A_enhanced = torch.tensor([indexed_tokens_A_enhanced])
            tokens_tensor_B_enhanced = torch.tensor([indexed_tokens_B_enhanced])


            # If you have a GPU, put everything on cuda
            tokens_tensor_A_enhanced = tokens_tensor_A_enhanced.to(device=device)
            tokens_tensor_B_enhanced = tokens_tensor_B_enhanced.to(device=device)


            model.to(device=device)

            # Predict all tokens
            total_logprobs_A_enhanced = 0
            total_logprobs_B_enhanced = 0

            with torch.no_grad():

                probs_A_enhanced = model(tokens_tensor_A_enhanced)  # , segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
                probs_B_enhanced = model(tokens_tensor_B_enhanced)  # , segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

                logprobs_A_enhanced = torch.nn.functional.log_softmax(probs_A_enhanced[0], dim=-1)
                logprobs_B_enhanced = torch.nn.functional.log_softmax(probs_B_enhanced[0], dim=-1)

                probs_array_A_enhanced = []
                probs_array_B_enhanced = []

                # A
                for index_item in masked_indices_items_A_text_enhanced:
                    index, item = index_item
                    total_logprobs_A_enhanced += logprobs_A_enhanced[0, index, item].item()
                    probs_array_A_enhanced = logprobs_A_enhanced[
                        0, torch.arange(len(indexed_tokens_A_enhanced)), indexed_tokens_A_enhanced]

                # B
                for index_item in masked_indices_items_B_text_enhanced:
                    index, item = index_item
                    total_logprobs_B_enhanced += logprobs_B_enhanced[0, index, item].item()
                    probs_array_B_enhanced = logprobs_B_enhanced[
                        0, torch.arange(len(indexed_tokens_B_enhanced)), indexed_tokens_B_enhanced]

                # prob shift
                c = total_logprobs_A_enhanced / tokenized_option_A_len
                w = total_logprobs_B_enhanced / tokenized_option_B_len

                if correct_answer == 'B':
                    c, w = w, c

                description[current_alt]['correct']['ans'].append(c)
                description[current_alt]['wrong']['ans'].append(w)
                indices[current_alt]['ans'].append(q_index)

                if discrim_word:
                    c = probs_array_A_enhanced[discrim_word_index_enhanced_A].item()
                    w = probs_array_B_enhanced[discrim_word_index_enhanced_B].item()

                    if correct_answer == 'B':
                        c, w = w, c

                    description[current_alt]['correct']['dis'].append(c)
                    description[current_alt]['wrong']['dis'].append(w)
                    indices[current_alt]['dis'].append(q_index)
                else:
                    description[current_alt]['correct']['dis'].append(None)
                    description[current_alt]['wrong']['dis'].append(None)
                    # indices[current_alt]['dis'].append(q_index)

                max_index_enhanced = np.argmax([total_logprobs_A_enhanced / tokenized_option_A_len, total_logprobs_B_enhanced
                                                / tokenized_option_B_len ])

                prediction_enhanced = "A" if max_index_enhanced == 0 else "B"


                if prediction_enhanced == correct_answer .strip().strip('.').replace(' ', ''):
                    answers[current_alt].append(1)
                    correct_preds_enhanced += 1

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

    accuracy_enhanced = correct_preds_enhanced / all_preds
    counts[current_alt]['all'] = all_preds
    print("accuracy: {}/{} = {}".format(correct_preds_enhanced, all_preds, accuracy_enhanced))
    print("stability: {}/{} = {}%".format(stability_match, all_preds, stability_match / all_preds))
    description[current_alt]['accuracy'] = accuracy_enhanced
    description[current_alt]['stability'] = stability_match / all_preds

    # print(description)
with open('description_dump_roberta.pickle', 'wb') as f:
    pickle.dump((description, indices, answers, counts, accuracies, stabilities), f)

