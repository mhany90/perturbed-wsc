import torch
from pytorch_transformers import RobertaModel, RobertaTokenizer, RobertaForMaskedLM
import numpy as np
from copy import deepcopy
import re
import pandas as pd


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

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

correct_preds = 0
correct_preds_enhanced = 0
stability_match  = 0

all_preds = 0

# Load pre-trained model (weights)
model = RobertaForMaskedLM.from_pretrained('roberta-large')
model.eval()

for q_index, dp_split in wsc_datapoints.iterrows():
    if dp_split['text_adverb'].replace(' ', '') != '-' and dp_split['text_adverb'].replace(' ', ''):

        # Tokenized input
        correct_answer = dp_split['correct_answer']

        #check for empty
        text = dp_split['text_original'].strip().lower()
        text = re.sub(' +', ' ', text)
        print(text, " text")

        text_enhanced = dp_split['text_adverb'].lower()
        text_enhanced = re.sub(' +', ' ', text_enhanced)


        tokenized_text = tokenizer.encode(text, add_special_tokens=True)
        tokenized_enhanced_text = tokenizer.encode(text_enhanced, add_special_tokens=True)

        tokens_pre_word_piece_A = dp_split['answer_a'].strip().lower()
        tokens_pre_word_piece_B = dp_split['answer_b'].strip().lower()
        print(tokens_pre_word_piece_A , " tokens_pre_word_piece_A ")
        print(tokens_pre_word_piece_B , " tokens_pre_word_piece_B ")


        pronoun = 'because ' + dp_split['pron'].lower()
        print(pronoun, "pronoun")
        pronoun_index_orig =  int(dp_split['pron_index'])
        pronoun_index_orig_enhanced =  int(dp_split['pron_index_adverb'])

        tokenized_option_A = tokenizer.encode(tokens_pre_word_piece_A, add_special_tokens=True)[1:-1]
        tokenized_option_B = tokenizer.encode(tokens_pre_word_piece_B, add_special_tokens=True)[1:-1]
        tokenized_pronoun = tokenizer.encode(pronoun, add_special_tokens=True)
        print(tokenized_pronoun, "tokenized_pronoun")

        tokenized_option_A_len = len(tokenized_option_A)
        tokenized_option_B_len = len(tokenized_option_B)

        print(tokenized_option_A, "tokenized_option A")
        print(tokenized_option_B, "tokenized_option B")

        matched_pronouns_text = find_sub_list([tokenized_pronoun[-2]], tokenized_text)
        matched_pronouns_enhanced_text = find_sub_list([tokenized_pronoun[-2]],  tokenized_enhanced_text)

        first_indices_text = np.array([mp[0] for mp in matched_pronouns_text])
        first_indices_text_enhanced = np.array([mp[0] for mp in matched_pronouns_enhanced_text])

        print(matched_pronouns_text, "matched_pronouns_text")
        print(matched_pronouns_enhanced_text, "matched_pronouns_text_enhanced")

        correct_idx_text = (np.abs(first_indices_text - pronoun_index_orig)).argmin()
        correct_idx_text_enhanced = (np.abs(first_indices_text_enhanced - pronoun_index_orig_enhanced)).argmin()
        print(correct_idx_text_enhanced, " correct_idx_text_enhanced")

        pronoun_index_text = matched_pronouns_text[correct_idx_text][0]
        pronoun_index_text_enhanced  = matched_pronouns_enhanced_text[correct_idx_text_enhanced][0]
        print(pronoun_index_text_enhanced, " pronoun_index_text_enhanced")

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

        #get index item
        masked_indices_items_A_text = [(index, item) for index, item in
                                      zip(range(masked_indices_A_text[0],masked_indices_A_text[1] + 1),tokenized_option_A)]
        masked_indices_items_A_text_enhanced = [(index, item) for index, item in
                                       zip(range(masked_indices_A_text[0], masked_indices_A_text_enhanced[1] +1 ),tokenized_option_A)]

        masked_indices_items_B_text = [(index, item) for index, item in
                                       zip(range(masked_indices_A_text[0], masked_indices_B_text[1] + 1),tokenized_option_B)]
        masked_indices_items_B_text_enhanced = [(index, item) for index, item in
                                                zip(range(masked_indices_B_text[0], masked_indices_B_text_enhanced[1] + 1),
                                                    tokenized_option_B)]


        for masked_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
            tokenized_text_A[masked_index] = tokenizer.encode('<mask>')[0]
        print(tokenized_text_A, "tokenized_text A MASKED")

        for masked_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
            tokenized_text_enhanced_A[masked_index] = tokenizer.encode('<mask>')[0]
        print(tokenized_text_enhanced_A, "tokenized_enchanced_text A MASKED")

        for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
            tokenized_text_B[masked_index] = tokenizer.encode('<mask>')[0]
        print(tokenized_text_B, "tokenized_text B MASKED")

        for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
            tokenized_text_enhanced_B[masked_index] = tokenizer.encode('<mask>')[0]
        print(tokenized_text_enhanced_B, "tokenized_enchanced_text B MASKED")


        # Convert token to vocabulary indices
        indexed_tokens_A = tokenized_text_A #tokenizer.encode(' '.join(tokenized_text_A), add_special_tokens=True)
        indexed_tokens_B = tokenized_text_B  #tokenizer.encode(' '.join(tokenized_text_B), add_special_tokens=True)

        #enhanced
        indexed_tokens_A_enhanced = tokenized_text_enhanced_A #tokenizer.encode(' '.join(tokenized_text_enhanced_A), add_special_tokens=True)
        indexed_tokens_B_enhanced = tokenized_text_enhanced_B #tokenizer.encode(' '.join(tokenized_text_enhanced_B), add_special_tokens=True)


        # Convert inputs to PyTorch tensors
        tokens_tensor_A = torch.tensor([indexed_tokens_A])
        tokens_tensor_B = torch.tensor([indexed_tokens_B])

        #enhanced
        tokens_tensor_A_enhanced = torch.tensor([indexed_tokens_A_enhanced])
        tokens_tensor_B_enhanced = torch.tensor([indexed_tokens_B_enhanced])


        # If you have a GPU, put everything on cuda
        tokens_tensor_A = tokens_tensor_A.to(device=device)
        tokens_tensor_B = tokens_tensor_B.to(device=device)


        model.to(device=device)

        # Predict all tokens
        total_logprobs_A = 0
        total_logprobs_B = 0

        total_logprobs_A_enhanced = 0
        total_logprobs_B_enhanced = 0

        with torch.no_grad():
            probs_A = model(tokens_tensor_A)#, segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
            probs_B = model(tokens_tensor_B)#, segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

            probs_A_enhanced = model(tokens_tensor_A_enhanced)  # , segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
            probs_B_enhanced = model(tokens_tensor_B_enhanced)  # , segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

            logprobs_A = torch.nn.functional.log_softmax(probs_A[0], dim=-1)
            logprobs_B = torch.nn.functional.log_softmax(probs_B[0], dim=-1)
            print(logprobs_A.shape, "logprobs_A")

            logprobs_A_enhanced = torch.nn.functional.log_softmax(probs_A_enhanced[0], dim=-1)
            logprobs_B_enhanced = torch.nn.functional.log_softmax(probs_B_enhanced[0], dim=-1)

            print("-----------A---------------")

            for index_item in masked_indices_items_A_text:
                index, item = index_item
                print(index, tokenizer.decode(item), " : index, item")
                #print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_A +=  logprobs_A[0,index,item].item()

            for index_item in masked_indices_items_A_text_enhanced:
                index, item = index_item
                print(index, tokenizer.decode(item), " : index, item")
                # print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_A_enhanced += logprobs_A_enhanced[0, index, item].item()

            print("-----------B---------------")

            for index_item in masked_indices_items_B_text:
                index, item = index_item
                print(index, tokenizer.decode(item), " : index, item")
                #print(probs_A[0, index, item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_B += logprobs_B[0,index,item].item()

            for index_item in masked_indices_items_B_text_enhanced:
                index, item = index_item
                print(index, tokenizer.decode(item), " : index, item")
                # print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_B_enhanced += logprobs_B_enhanced[0, index, item].item()


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