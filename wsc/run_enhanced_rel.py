import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import deepcopy
import pandas as pd

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

correct_preds = 0
correct_preds_enhanced = 0
stability_match  = 0

all_preds = 0

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

#for dp in wsc_datapoints[1:]:
for q_index, dp_split in wsc_datapoints.iterrows():
    if dp_split['text_adverb'].replace(' ', '') != '-' and dp_split['text_adverb'].replace(' ', ''):
        # Tokenized input
        correct_answer = dp_split['correct_answer']

        #check for empty
        text = "[CLS] " + dp_split['text_original']  + " [SEP]"
        text_enhanced = "[CLS] " + dp_split['text_adverb']  + " [SEP]"

        tokenized_text = tokenizer.tokenize(text)
        tokenized_enhanced_text = tokenizer.tokenize(text_enhanced)

        tokens_pre_word_piece_A = dp_split['answer_a']
        tokens_pre_word_piece_B = dp_split['answer_b']

        pronoun = dp_split['pron'].strip()
        pronoun_index_orig =  int(dp_split['pron_index'])
        pronoun_index_orig_enhanced =  int(dp_split['pron_index_adverb'])

        tokenized_option_A = tokenizer.tokenize(tokens_pre_word_piece_A)
        tokenized_option_B = tokenizer.tokenize(tokens_pre_word_piece_B)
        tokenized_pronoun = tokenizer.tokenize(pronoun)

        tokenized_option_A_len = len(tokenized_option_A)
        tokenized_option_B_len = len(tokenized_option_B)

        print(tokenized_option_A, "tokenized_option A")
        print(tokenized_option_B, "tokenized_option B")

        matched_pronouns_text = find_sub_list(tokenized_pronoun, tokenized_text)
        matched_pronouns_enhanced_text = find_sub_list(tokenized_pronoun,  tokenized_enhanced_text)

        first_indices_text = np.array([mp[0] for mp in matched_pronouns_text])
        first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])

        print(matched_pronouns_text, "matched_pronouns_text")
        print(matched_pronouns_enhanced_text, "matched_pronouns_text_enhanced")

        correct_idx_text = (np.abs(first_indices_text - pronoun_index_orig)).argmin()
        correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
        print(correct_idx_text_enhanced, " correct_idx_text_enhanced")

        pronoun_index_text = matched_pronouns_text[correct_idx_text][0]
        pronoun_index_text_enhanced  = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]

        tokenized_text_A = replace_pronoun(tokenized_text, pronoun_index_text, tokenized_option_A)
        tokenized_text_B = replace_pronoun(tokenized_text, pronoun_index_text, tokenized_option_B)

        tokenized_text_enhanced_A = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_A)
        tokenized_text_enhanced_B = replace_pronoun(tokenized_enhanced_text, pronoun_index_text_enhanced, tokenized_option_B)

        print(tokenized_text_A, "tokenized_text_A")
        print(tokenized_text_enhanced_A, "tokenized_text_enhanced_A")

        matched_A_text = find_sub_list(tokenized_option_A, tokenized_text_A)
        matched_B_text = find_sub_list(tokenized_option_B, tokenized_text_B)

        matched_A_text_enhanced = find_sub_list(tokenized_option_A, tokenized_text_enhanced_A)
        matched_B_text_enhanced = find_sub_list(tokenized_option_B, tokenized_text_enhanced_B)

        print(matched_A_text, "matched A")
        print(matched_A_text_enhanced, "matched A enhanced")

        masked_indices_A_text = [m for m in matched_A_text if m[0] == pronoun_index_text][0]
        masked_indices_A_text_enhanced = [m for m in matched_A_text_enhanced if m[0] == pronoun_index_text_enhanced][0]

        masked_indices_B_text = [m for m in matched_B_text if m[0] == pronoun_index_text][0]
        masked_indices_B_text_enhanced = [m for m in matched_B_text_enhanced if m[0] == pronoun_index_text_enhanced][0]


        tokenized_text_A_pre_mask = deepcopy(tokenized_text_A)
        tokenized_text_B_pre_mask = deepcopy(tokenized_text_B)

        tokenized_text_A_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_A)
        tokenized_text_B_pre_mask_enhanced = deepcopy(tokenized_text_enhanced_B)

        #copies
        copies_A = []
        copies_A_masked = []
        copies_B = []
        copies_B_masked = []

        copies_A_enhanced = []
        copies_A_enhanced_masked = []

        copies_B_enhanced = []
        copies_B_enhanced_masked = []

        #copies
        for masked_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
            copy_A = deepcopy(tokenized_text_A)
            copies_A.append(copy_A)
            copy_A_enhanced = deepcopy(tokenized_text_enhanced_A)
            copies_A_enhanced.append(copy_A_enhanced)


        for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
            copy_B = deepcopy(tokenized_text_B)
            copies_B.append(copy_B)
            copy_B_enhanced = deepcopy(tokenized_text_enhanced_B)
            copies_B_enhanced.append(copy_B_enhanced)

        for masked_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
            tokenized_text_A[masked_index] = '[MASK]'
            for c in range(len(copies_A)):
                copies_A[c][masked_index] = '[MASK]'
                copies_A_enhanced[c][masked_index] = '[MASK]'
            last_list = copies_A.pop(-1)
            last_list_enhanced = copies_A_enhanced.pop(-1)
            copies_A_masked.append(last_list)
            copies_A_enhanced_masked.append(last_list_enhanced)

        copies_A_masked.reverse()
        copies_A_enhanced_masked.reverse()

        for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
            tokenized_text_B[masked_index] = '[MASK]'
            for c in range(len(copies_B)):
                copies_B[c][masked_index] = '[MASK]'
                copies_B_enhanced[c][masked_index] = '[MASK]'
            last_list = copies_B.pop(-1)
            last_list_enhanced = copies_B_enhanced.pop(-1)
            copies_B_masked.append(last_list)
            copies_B_enhanced_masked.append(last_list_enhanced)

        copies_B_masked.reverse()
        copies_B_enhanced_masked.reverse()

        """
        print(tokenized_text_A, "tokenized_text A MASKED")
        
        for masked_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
            tokenized_text_enhanced_A[masked_index] = '[MASK]'
        print(tokenized_text_enhanced_A, "tokenized_enchanced_text A MASKED")

        for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
            tokenized_text_B[masked_index] = '[MASK]'
        print(tokenized_text_B, "tokenized_text B MASKED")

        for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
            tokenized_text_enhanced_B[masked_index] = '[MASK]'
        print(tokenized_text_enhanced_B, "tokenized_enchanced_text B MASKED")
        """

        masked_lm_labels_A = []
        masked_lm_labels_B = []
        masked_lm_labels_A_enhanced = []
        masked_lm_labels_B_enhanced = []

        # Convert token to vocabulary indices
        indexed_tokens_A_copies = []
        indexed_tokens_enhanced_A_copies = []
        indexed_tokens_B_copies = []
        indexed_tokens_enhanced_B_copies = []


        for c in copies_A_masked:
            indexed_tokens_A = tokenizer.convert_tokens_to_ids(c)
            indexed_tokens_A_copies.append(indexed_tokens_A)

        for c in copies_A_enhanced_masked:
            indexed_tokens_A_enhanced = tokenizer.convert_tokens_to_ids(c)
            indexed_tokens_enhanced_A_copies.append(indexed_tokens_A_enhanced)

        for c in copies_B_masked:
            indexed_tokens_B = tokenizer.convert_tokens_to_ids(c)
            indexed_tokens_B_copies.append(indexed_tokens_B)

        for c in copies_B_enhanced_masked:
            indexed_tokens_B_enhanced = tokenizer.convert_tokens_to_ids(c)
            indexed_tokens_enhanced_B_copies.append(indexed_tokens_B_enhanced)

        indexed_tokens_A_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask)
        indexed_tokens_B_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask)

        #enhanced
        indexed_tokens_A_pre_mask_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask_enhanced)
        indexed_tokens_B_pre_mask_enhanced = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask_enhanced)

        #options
        indexed_tokens_option_A = tokenizer.convert_tokens_to_ids(tokenized_option_A)
        indexed_tokens_option_B = tokenizer.convert_tokens_to_ids(tokenized_option_B)

        #mask all labels but wsc options
        for token_index in range(len(copies_A_masked[-1])):
            if token_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
                masked_lm_labels_A.append(indexed_tokens_A_pre_mask[token_index])
            else:
                masked_lm_labels_A.append(-1)

        #mask all labels but wsc options
        for token_index in range(len(copies_B_masked[-1])):
            if token_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
                masked_lm_labels_B.append(indexed_tokens_B_pre_mask[token_index])
            else:
                masked_lm_labels_B.append(-1)

        # mask all labels but wsc options (enhanced)
        for token_index in range(len(copies_A_enhanced_masked[-1])):
            if token_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
                masked_lm_labels_A_enhanced.append(indexed_tokens_A_pre_mask_enhanced[token_index])
            else:
                masked_lm_labels_A_enhanced.append(-1)

        # mask all labels but wsc options
        for token_index in range(len(copies_B_enhanced_masked[-1])):
            if token_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
                masked_lm_labels_B_enhanced.append(indexed_tokens_B_pre_mask_enhanced[token_index])
            else:
                masked_lm_labels_B_enhanced.append(-1)


        masked_tokens_A =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A if i!=-1]))
        masked_tokens_B =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B if i!=-1]))

        masked_tokens_A_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A_enhanced if i != -1]))
        masked_tokens_B_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B_enhanced if i != -1]))

        masked_lm_labels_A_non_neg = [(index, item) for index, item in enumerate(masked_lm_labels_A) if item!=-1]
        masked_lm_labels_B_non_neg =  [(index, item) for index, item in enumerate(masked_lm_labels_B) if item!=-1]

        masked_lm_labels_A_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_A_enhanced) if item != -1]
        masked_lm_labels_B_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_B_enhanced) if item != -1]

        # Convert inputs to PyTorch tensors
        tokens_tensor_A_copies = [torch.tensor([indexed_tokens_A]) for indexed_tokens_A in indexed_tokens_A_copies]
        tokens_tensor_B_copies = [torch.tensor([indexed_tokens_B]) for indexed_tokens_B in indexed_tokens_B_copies]

        tokens_tensor_option_A = torch.tensor([indexed_tokens_option_A])
        tokens_tensor_option_B = torch.tensor([indexed_tokens_option_B])

        #enhanced
        tokens_tensor_A_copies_enhanced = [torch.tensor([indexed_tokens_A_enhanced]).to(device=device) for indexed_tokens_A_enhanced in indexed_tokens_enhanced_A_copies]
        tokens_tensor_B_copies_enhanced = [torch.tensor([indexed_tokens_B_enhanced]).to(device=device) for indexed_tokens_B_enhanced in indexed_tokens_enhanced_B_copies]

        model.to(device=device)

        # Predict all tokens
        total_logprobs_A = 0
        total_logprobs_B = 0

        total_logprobs_A_enhanced = 0
        total_logprobs_B_enhanced = 0

        prob_copies_A = []
        prob_copies_enhanced_A = []

        prob_copies_B = []
        prob_copies_enhanced_B = []

        with torch.no_grad():
            for c in tokens_tensor_A_copies:
                print(c, " c")
                probs_A = model(c)
                prob_copies_A.append(probs_A)

            for c in tokens_tensor_A_copies_enhanced:
                probs_A_enhanced = model(c)
                prob_copies_enhanced_A.append(probs_A_enhanced)

            for c in tokens_tensor_B_copies:
                probs_B = model(c)
                prob_copies_B.append(probs_B)

            for c in tokens_tensor_B_copies_enhanced:
                probs_B_enhanced = model(c)
                prob_copies_enhanced_B.append(probs_B_enhanced)

            print(len(prob_copies_A), " len prob_copies_A")
            logprobs_A = [torch.nn.functional.log_softmax(p_A, dim=-1) for p_A in prob_copies_A]
            logprobs_B = [torch.nn.functional.log_softmax(p_B, dim=-1) for p_B in prob_copies_B]

            logprobs_A_enhanced = [torch.nn.functional.log_softmax(p_A_enhanced, dim=-1) for p_A_enhanced in prob_copies_enhanced_A]
            logprobs_B_enhanced = [torch.nn.functional.log_softmax(p_B_enhanced, dim=-1) for p_B_enhanced in prob_copies_enhanced_B]


            print("-----------A---------------")
            print(len(logprobs_A), " len logprobs_A")
            print(len(masked_lm_labels_A_non_neg), "len masked_lm_labels_A_non_neg")
            for copy_ixd, index_item in enumerate(masked_lm_labels_A_non_neg[::-1]):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                total_logprobs_A +=  logprobs_A[copy_ixd][0,index,item].item()

            for copy_ixd, index_item in  enumerate(masked_lm_labels_A_non_neg_enhanced[::-1]):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                total_logprobs_A_enhanced += logprobs_A_enhanced[copy_ixd][0, index, item].item()


            print("-----------B---------------")

            for copy_ixd, index_item in enumerate(masked_lm_labels_B_non_neg[::-1]):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                total_logprobs_B +=  logprobs_B[copy_ixd][0,index,item].item()

            for copy_ixd, index_item in  enumerate(masked_lm_labels_B_non_neg_enhanced[::-1]):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                total_logprobs_B_enhanced += logprobs_B_enhanced[copy_ixd][0, index, item].item()

            print(total_logprobs_A / tokenized_option_A_len  , " total_probs_A / tokenized_option_A_len")
            print(total_logprobs_B / tokenized_option_B_len , " total_probs_B / tokenized_option_B_len")
            print(correct_answer.strip().strip('.').replace(' ', ''), " correct_answer")

            print(total_logprobs_A_enhanced / tokenized_option_A_len, " total_probs_A / tokenized_option_A_len")
            print(total_logprobs_B_enhanced / tokenized_option_B_len, " total_probs_B / tokenized_option_B_len")
            print(correct_answer.strip().strip('.').replace(' ', ''), " correct_answer")

            max_index = np.argmax([total_logprobs_A / tokenized_option_A_len, total_logprobs_B / tokenized_option_B_len ])
            max_index_enhanced = np.argmax([total_logprobs_A_enhanced / tokenized_option_A_len, total_logprobs_B_enhanced
                                            / tokenized_option_B_len ])

            prediction = "A" if max_index == 0 else "B"
            prediction_enhanced = "A" if max_index_enhanced == 0 else "B"

            print(prediction, " prediction")
            print(prediction_enhanced, " prediction enhanced")

            if prediction == correct_answer.strip().strip('.').replace(' ', ''):
                correct_preds += 1
            if prediction_enhanced == correct_answer .strip().strip('.').replace(' ', ''):
                correct_preds_enhanced += 1
            if prediction_enhanced == prediction:
                stability_match += 1

            all_preds += 1
            print("#############################################################################")
    else:
        continue

accuracy = correct_preds/all_preds
print(all_preds, " : all_preds")
print(correct_preds, " : correct_preds")
print(accuracy, " : accuracy")

accuracy_enhanced = correct_preds_enhanced/all_preds
print(all_preds, " : all_preds")
print(correct_preds_enhanced, " : correct_preds enhanced")
print(accuracy_enhanced, " : accuracy_enhancedy")

print(stability_match, ": stability_match")
print(stability_match / all_preds , ": stability_match %")