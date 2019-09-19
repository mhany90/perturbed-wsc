import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import deepcopy

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/new_test.tsv'
wsc_file = open(path_to_wsc, 'r')
wsc_datapoints = wsc_file.readlines()

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

correct_preds = 0
all_preds = 0

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

for dp in wsc_datapoints[1:274]:
    dp_split = dp.split('\t')
    # Tokenized input
    correct_answer = dp_split[-2]
    source = dp_split[-1]

    text_A = "[CLS] " + dp_split[3] + "[SEP]"
    text_B = "[CLS] " + dp_split[6]  + "[SEP]"
    text = "[CLS] " + dp_split[0]  + "[SEP]"




    text_A_split = text_A.split()
    text_B_split = text_B.split()

    tokenized_text_A = tokenizer.tokenize(text_A)
    tokenized_text_B = tokenizer.tokenize(text_B)
    tokenized_text = tokenizer.tokenize(text)


    print(tokenized_text_A, "tokenized_text A")
    print(tokenized_text_B, "tokenized_text B")

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    #indices_str_A = dp_split[4].strip()
    #indices_str_B = dp_split[6].strip()

    #index1_A = indices_str_A.split(',')[0].strip('(')
    #index2_A = indices_str_A.split(',')[1].strip(')')

    #index1_B = indices_str_B.split(',')[0].strip('(')
    #index2_B = indices_str_B.split(',')[1].strip(')')

    #masked_indices_before_tok_A = (int(index1_A) + 1 , int(index1_A) + 3)
    #masked_indices_before_tok_B = (int(index1_B) + 1 , int(index1_B) + 3)

    #print(masked_indices_before_tok_A, "masked_indices_before_tok A")
    #print(masked_indices_before_tok_B, "masked_indices_before_tok B")

    #tokens_pre_word_piece_A = ' '.join(text_A_split[masked_indices_before_tok_A[0]:masked_indices_before_tok_A[1]])
    #tokens_pre_word_piece_B = ' '.join(text_B_split[masked_indices_before_tok_B[0]:masked_indices_before_tok_B[1]])

    tokens_pre_word_piece_A = dp_split[5]
    tokens_pre_word_piece_B = dp_split[8]
    pronoun = dp_split[1].strip()
    pronoun_index_orig =  int(dp_split[2].strip())

    tokenized_option_A = tokenizer.tokenize(tokens_pre_word_piece_A)
    tokenized_option_B = tokenizer.tokenize(tokens_pre_word_piece_B)
    tokenized_pronoun = tokenizer.tokenize(pronoun)


    tokenized_option_A_len = len(tokenized_option_A)
    tokenized_option_B_len = len(tokenized_option_B)


    print(tokenized_option_A, "tokenized_option A")
    print(tokenized_option_B, "tokenized_option B")

    matched_A = find_sub_list(tokenized_option_A, tokenized_text_A)
    matched_B = find_sub_list(tokenized_option_B, tokenized_text_B)
    matched_pronouns = find_sub_list(tokenized_pronoun, tokenized_text)
    first_indices = np.array([mp[0] for mp in matched_pronouns])
    print(matched_pronouns, "matched_pronouns")
    correct_idx = (np.abs(first_indices - pronoun_index_orig)).argmin()

    pronoun_index = matched_pronouns[correct_idx][0]
    print(pronoun_index)

    #print(matched_A, "matched A")
    masked_indices_A = [m for m in matched_A if m[0] == pronoun_index][0]
    #print(matched_B, "matched B")
    masked_indices_B = [m for m in matched_B if m[0] == pronoun_index][0]

    tokenized_text_A_pre_mask = deepcopy(tokenized_text_A)
    tokenized_text_B_pre_mask = deepcopy(tokenized_text_B)

    for masked_index in range(masked_indices_A[0], masked_indices_A[1]):
        tokenized_text_A[masked_index] = '[MASK]'
    print(tokenized_text_A, "tokenized_text A MASKED")

    for masked_index in range(masked_indices_B[0], masked_indices_B[1]):
        tokenized_text_B[masked_index] = '[MASK]'
    print(tokenized_text_B, "tokenized_text B MASKED")

    masked_lm_labels_A = []
    masked_lm_labels_B = []

    # Convert token to vocabulary indices
    indexed_tokens_A = tokenizer.convert_tokens_to_ids(tokenized_text_A)
    indexed_tokens_B = tokenizer.convert_tokens_to_ids(tokenized_text_B)
    indexed_tokens_A_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask)
    indexed_tokens_B_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask)

    #mask all labels but wsc options
    for token_index in range(len(indexed_tokens_A)):
        if token_index in range(masked_indices_A[0], masked_indices_A[1]):
            masked_lm_labels_A.append(indexed_tokens_A_pre_mask[token_index])
        else:
            masked_lm_labels_A.append(-1)

    #mask all labels but wsc options
    for token_index in range(len(indexed_tokens_B)):
        if token_index in range(masked_indices_B[0], masked_indices_B[1]):
            masked_lm_labels_B.append(indexed_tokens_B_pre_mask[token_index])
        else:
            masked_lm_labels_B.append(-1)

    #print(masked_lm_labels_A, "masked_lm_labels_A")
    #print(masked_lm_labels_B, "masked_lm_labels_B")

    #masked_tokens_A = ' '.join(tokenizer.convert_ids_to_tokens(masked_lm_labels_A))
    #masked_tokens_B = ' '.join(tokenizer.convert_ids_to_tokens(masked_lm_labels_B))
    masked_tokens_A =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A if i!=-1]))
    masked_tokens_B =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B if i!=-1]))

    #assert masked_tokens_A == ' '.join(tokenized_option_A)
    #assert masked_tokens_B == ' '.join(tokenized_option_B)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids_A = [0] * len(indexed_tokens_A)
    segments_ids_B = [0] * len(indexed_tokens_B)

    masked_lm_labels_A_non_neg = [(index, item) for index, item in enumerate(masked_lm_labels_A) if item!=-1]
    masked_lm_labels_B_non_neg =  [(index, item) for index, item in enumerate(masked_lm_labels_B) if item!=-1]

    # Convert inputs to PyTorch tensors
    tokens_tensor_A = torch.tensor([indexed_tokens_A])
    segments_tensors_A = torch.tensor([segments_ids_A])
    masked_lm_labels_A = torch.tensor([masked_lm_labels_A])

    tokens_tensor_B = torch.tensor([indexed_tokens_B])

    segments_tensors_B = torch.tensor([segments_ids_B])
    masked_lm_labels_B = torch.tensor([masked_lm_labels_B])

    # If you have a GPU, put everything on cuda
    tokens_tensor_A = tokens_tensor_A.to(device=device)
    tokens_tensor_B = tokens_tensor_B.to(device=device)

    segments_tensors_A = segments_tensors_A.to(device=device)
    segments_tensors_B = segments_tensors_B.to(device=device)


    masked_lm_labels_A =  masked_lm_labels_A.to(device=device)
    masked_lm_labels_B =  masked_lm_labels_B.to(device=device)
    #print(masked_lm_labels_A, " masked_lm_labels_A tensor")

    model.to(device=device)

    # Predict all tokens
    total_logprobs_A = 0
    total_logprobs_B = 0

    with torch.no_grad():
        probs_A = model(tokens_tensor_A)#, segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
        probs_B = model(tokens_tensor_B)#, segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

        logprobs_A = torch.nn.functional.log_softmax(probs_A, dim=-1)
        logprobs_B = torch.nn.functional.log_softmax(probs_B, dim=-1)

        print("-----------A---------------")

        for index_item in masked_lm_labels_A_non_neg:
            index, item = index_item
            print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
            #print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
            total_logprobs_A +=  logprobs_A[0,index,item].item()

        print("-----------B---------------")

        for index_item in masked_lm_labels_B_non_neg:
            index, item = index_item
            print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
            #print(probs_A[0, index, item].item(), " : probs_A[0,index,item].item()")
            total_logprobs_B += logprobs_B[0,index,item].item()



        print(total_logprobs_A / tokenized_option_A_len  , " total_probs_A / tokenized_option_A_len")
        print(total_logprobs_B / tokenized_option_B_len , " total_probs_B / tokenized_option_B_len")
        print(correct_answer.strip().strip('.').replace(' ', ''), " correct_answer")

        max_index = np.argmax([total_logprobs_A / tokenized_option_A_len, total_logprobs_B / tokenized_option_B_len ])
        prediction = "A" if max_index == 0 else "B"
        print(prediction, " prediction")
        if prediction == correct_answer.strip().strip('.').replace(' ', ''):
            correct_preds += 1
        all_preds += 1
        print("#############################################################################")

accuracy = correct_preds/all_preds
print(all_preds, " : all_preds")
print(correct_preds, " : correct_preds")
print(accuracy, " : accuracy")