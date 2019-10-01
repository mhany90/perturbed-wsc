import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
import sys

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


model_name = 'bert-base-uncased' if sys.argv[1] == '--debug' else 'bert-large-uncased'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(model_name)

# perturbation: correct/wrong: original/altered
# this dupplicates original but whatever the fuck
description = {}
indices = {}
answers = {}
attentions = {}
prediction_original = []
baseline_attentions = []
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained(model_name, output_attentions=True)
model.eval()
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

    correct_preds_enhanced, stability_match = 0, 0
    all_preds = 0

    print(current_alt)
    for q_index, dp_split in wsc_datapoints.iterrows():
        if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):
            # save the index
            # Tokenized input
            correct_answer = dp_split['correct_answer'].strip().strip('.').replace(' ', '')
            text_enhanced = "[CLS] " + dp_split[current_alt]  + " [SEP]"

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

            if current_alt == 'text_number':
                tokenized_pronoun = tokenizer.tokenize(dp_split['pron_number'].strip())
            elif current_alt == 'text_gender':
                tokenized_pronoun = tokenizer.tokenize(dp_split['pron_gender'].strip())

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
                tokenized_text_enhanced_A[masked_index] = '[MASK]'

            for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                tokenized_text_enhanced_B[masked_index] = '[MASK]'

            masked_lm_labels_A_enhanced = []
            masked_lm_labels_B_enhanced = []

            #enhanced
            indexed_tokens_A_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_A)
            indexed_tokens_B_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_B)
            indexed_tokens_A_pre_mask_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask_enhanced)
            indexed_tokens_B_pre_mask_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask_enhanced)

            #text with pron
            indexed_tokens_enhanced_with_pron = tokenizer.convert_tokens_to_ids(tokenized_enhanced_text)

            len_tokens_A_enhanced = len(indexed_tokens_A_enhanced)
            len_tokens_B_enhanced = len(indexed_tokens_B_enhanced)

            # mask all labels but wsc options (enhanced)
            for token_index in range(len_tokens_A_enhanced):
                if token_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
                    masked_lm_labels_A_enhanced.append(indexed_tokens_A_pre_mask_enhanced[token_index])
                else:
                    masked_lm_labels_A_enhanced.append(-1)

            # mask all labels but wsc options
            for token_index in range(len_tokens_B_enhanced):
                if token_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                    masked_lm_labels_B_enhanced.append(indexed_tokens_B_pre_mask_enhanced[token_index])
                else:
                    masked_lm_labels_B_enhanced.append(-1)

            masked_tokens_A_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A_enhanced if i != -1]))
            masked_tokens_B_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B_enhanced if i != -1]))

            masked_lm_labels_A_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_A_enhanced) if item != -1]
            masked_lm_labels_B_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_B_enhanced) if item != -1]

            tokens_tensor_A_enhanced = torch.tensor([indexed_tokens_A_enhanced])
            tokens_tensor_A_pre_mask = torch.tensor([indexed_tokens_A_pre_mask_enhanced])
            masked_lm_labels_A_enhanced = torch.tensor([masked_lm_labels_A_enhanced])

            tokens_tensor_B_enhanced = torch.tensor([indexed_tokens_B_enhanced])
            tokens_tensor_B_pre_mask = torch.tensor([indexed_tokens_B_pre_mask_enhanced])
            masked_lm_labels_B_enhanced = torch.tensor([masked_lm_labels_B_enhanced])

            tokens_tensor_enhanced_with_pron = torch.tensor([indexed_tokens_enhanced_with_pron])

            # mask_id = torch.tensor(tokenizer.convert_tokens_to_ids(['[MASK]'])).to(device=device)
            # If you have a GPU, put everything on cuda
            tokens_tensor_A_enhanced = tokens_tensor_A_enhanced.to(device=device)
            tokens_tensor_B_enhanced = tokens_tensor_B_enhanced.to(device=device)
            tokens_tensor_A_pre_mask = tokens_tensor_A_pre_mask.to(device=device)
            tokens_tensor_B_pre_mask = tokens_tensor_B_pre_mask.to(device=device)
            masked_lm_labels_A_enhanced = masked_lm_labels_A_enhanced.to(device=device)
            masked_lm_labels_B_enhanced = masked_lm_labels_B_enhanced.to(device=device)
            tokens_tensor_enhanced_with_pron = tokens_tensor_enhanced_with_pron.to(device=device)

            model.to(device=device)

            total_logprobs_A_enhanced = 0
            total_logprobs_B_enhanced = 0

            with torch.no_grad():
                probs_A_enhanced = model(tokens_tensor_A_enhanced)[0]
                probs_B_enhanced = model(tokens_tensor_B_enhanced)[0]

                logprobs_A_enhanced = torch.nn.functional.log_softmax(probs_A_enhanced, dim=-1)
                logprobs_B_enhanced = torch.nn.functional.log_softmax(probs_B_enhanced, dim=-1)

                probs_array_A_enhanced = []
                probs_array_B_enhanced = []

                # A
                for index_item in masked_lm_labels_A_non_neg_enhanced:
                    index, item = index_item
                    total_logprobs_A_enhanced += logprobs_A_enhanced[0, index, item].item()
                    probs_array_A_enhanced = logprobs_A_enhanced[0, torch.arange(len(indexed_tokens_A_enhanced)), indexed_tokens_A_enhanced]

                # B
                for index_item in masked_lm_labels_B_non_neg_enhanced:
                    index, item = index_item
                    total_logprobs_B_enhanced += logprobs_B_enhanced[0, index, item].item()
                    probs_array_B_enhanced = logprobs_B_enhanced[0, torch.arange(len(indexed_tokens_B_enhanced)), indexed_tokens_B_enhanced]

                # prob shift
                c = total_logprobs_A_enhanced / tokenized_option_A_len
                w = total_logprobs_B_enhanced / tokenized_option_B_len

                if correct_answer == 'B':
                    c, w = w, c

                description[current_alt]['correct']['ans'].append(c)
                description[current_alt]['wrong']['ans'].append(w)
                indices[current_alt]['ans'].append(q_index)

                ## attention analysis

                interesting_phenomena_A = {}
                interesting_phenomena_B = {}
                interesting_phenomena_pron = {}

                interesting_phenomena_A['answer'] = [i[0] for i in masked_lm_labels_A_non_neg_enhanced]
                interesting_phenomena_B['answer'] = [i[0] for i in masked_lm_labels_B_non_neg_enhanced]

                interesting_phenomena_A['discrim'] = []
                interesting_phenomena_B['discrim'] = []
                interesting_phenomena_pron['discrim'] = []

                if discrim_word:
                    discrim_tensor = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(discrim_word))
                    for token in discrim_tensor:
                        interesting_phenomena_A['discrim'].append((tokens_tensor_A_pre_mask == token).nonzero()[0][1].item())
                        interesting_phenomena_B['discrim'].append((tokens_tensor_B_pre_mask == token).nonzero()[0][1].item())
                        interesting_phenomena_pron['discrim'].append((tokens_tensor_enhanced_with_pron == token).nonzero()[0][1].item())



                new_A_attn = model(tokens_tensor_A_pre_mask)[1]
                new_B_attn = model(tokens_tensor_B_pre_mask)[1]

                new_attn_from_pron = model(tokens_tensor_enhanced_with_pron)[1]

                # attention paid to discrim
                c, w = 0, 0
                c_p, w_p = 0, 0
                if discrim_word:
                    c = new_A_attn[-1][0, :, :, interesting_phenomena_A['discrim']].sum(dim=-1).sum(dim=-1).max(dim=-1)[0].item() #* len_tokens_A_enhanced
                    w = new_B_attn[-1][0, :, :, interesting_phenomena_B['discrim']].sum(dim=-1).sum(dim=-1).max(dim=-1)[0].item() #* len_tokens_A_enhanced

                    print(new_attn_from_pron[-1][0, :,interesting_phenomena_pron['discrim']  ,pronoun_index_text_enhanced].item())
                    a = new_attn_from_pron[-1][0, :, interesting_phenomena_pron['discrim'], pronoun_index_text_enhanced]\
                        .sum(dim=-1).mean(dim=0).item()  # * len_tokens_A_enhanced

                if correct_answer == 'B':
                    c, w = w, c

                description[current_alt]['correct']['attn'].append(c)
                description[current_alt]['wrong']['attn'].append(w)
                description[current_alt]['all']['pron_attn'].append(a)

                # # pick the
                # mask_new_A = [n for (n, i) in enumerate(tokens_tensor_A_pre_mask.squeeze()) if i in old_A_tokens]
                # mask_new_B = [n for (n, i) in enumerate(tokens_tensor_B_pre_mask.squeeze()) if i in old_B_tokens]
                # mask_old_A = [n for (n, i) in enumerate(old_A_tokens.squeeze()) if i in tokens_tensor_A_pre_mask]
                # mask_old_B = [n for (n, i) in enumerate(old_B_tokens.squeeze()) if i in tokens_tensor_A_pre_mask]
                #
                # new_A_attn = new_A_attn[-1][0, :, :, mask_new_A].sum(dim=-1).sum(dim=-1).mean(dim=-1)
                # old_A_attn = old_A_attn[-1][0, :, :, mask_old_A].sum(dim=-1).sum(dim=-1).mean(dim=-1)
                # new_B_attn = new_B_attn[-1][0, :, :, mask_new_B].sum(dim=-1).sum(dim=-1).mean(dim=-1)
                # old_B_attn = old_B_attn[-1][0, :, :, mask_old_B].sum(dim=-1).sum(dim=-1).mean(dim=-1)
                # diff_A = new_A_attn - old_A_attn
                # diff_B = new_B_attn - old_B_attn
                ## END

                if discrim_word and 1 == 0:
                    discrim_tensor = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(discrim_word))
                    mask_discrim_A = tokens_tensor_A_pre_mask.clone().to(device=device)
                    mask_discrim_B = tokens_tensor_B_pre_mask.clone().to(device=device)
                    for token in discrim_tensor:
                        tokens_tensor_A_pre_mask[tokens_tensor_A_pre_mask == token] = mask_id
                        tokens_tensor_B_pre_mask[tokens_tensor_B_pre_mask == token] = mask_id

                    indices_A = (tokens_tensor_A_pre_mask == mask_id).nonzero(as_tuple=True)
                    indices_B = (tokens_tensor_A_pre_mask == mask_id).nonzero(as_tuple=True)
                    items_A = mask_discrim_A[indices_A]
                    items_B = mask_discrim_B[indices_B]

                    try:
                        probs_A_discrim = torch.nn.functional.log_softmax(model(tokens_tensor_A_pre_mask), dim=-1)
                        probs_B_discrim = torch.nn.functional.log_softmax(model(tokens_tensor_B_pre_mask), dim=-1)
                    except:
                        indices_A, items_A, indices_B, items_B = [[], []], [], [[], []], []

                    total_logprobs_A_discrim = 0
                    total_logprobs_B_discrim = 0

                    for index, item in zip(indices_A[1], items_A):
                        total_logprobs_A_discrim += probs_A_discrim[0, index, item].item()

                    for index, item in zip(indices_B[1], items_B):
                        total_logprobs_B_discrim += probs_B_discrim[0, index, item].item()

                    c = total_logprobs_A_discrim
                    w = total_logprobs_B_discrim

                    if correct_answer == 'B':
                        c, w = w, c

                    if c != 0 and w != 0:
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

                if prediction_enhanced == correct_answer.strip().strip('.').replace(' ', ''):
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
    counts[current_alt]['all'] = all_preds

    accuracy_enhanced = correct_preds_enhanced/all_preds
    print("accuracy: {}/{} = {}".format(accuracies[current_alt]['all'], all_preds, accuracy_enhanced))
    print("stability: {}/{} = {}%".format(stabilities[current_alt]['all'], all_preds, stability_match / all_preds))

#print(description)
with open('description_dump_bert_headmaxnonorm_pronmean.pickle', 'wb') as f:
    pickle.dump((description, indices, answers, counts, accuracies, stabilities), f)
