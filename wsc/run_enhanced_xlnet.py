import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, XLNetLMHeadModel, XLNetTokenizer
import numpy as np
from copy import deepcopy

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import re
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.tsv'
wsc_file = open(path_to_wsc, 'r')
wsc_datapoints = wsc_file.readlines()

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

correct_preds = 0
correct_preds_enhanced = 0
stability_match  = 0

all_preds = 0

# Load pre-trained model (weights)
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
model.eval()

for dp in wsc_datapoints[1:]:
    dp_split = dp.split('\t')
    if dp_split[6].replace(' ', '') != '-' and dp_split[6].replace(' ', ''):

        # Tokenized input
        correct_answer = dp_split[-3]

        #check for empty
        """
        text = "<cls> " + dp_split[0].lower()  + " <sep>"
        text = re.sub(' +', ' ', text)
        text_enhanced = "<cls> " + dp_split[6].lower()  + " <sep>"
        text_enhanced = re.sub(' +', ' ', text_enhanced)
        """

        text = dp_split[0].lower()
        text = re.sub(' +', ' ', text)
        text_enhanced =  dp_split[6].lower()
        text_enhanced = re.sub(' +', ' ', text_enhanced)
        print(text, "text")

        tokenized_text = tokenizer.tokenize(text)
        tokenized_enhanced_text = tokenizer.tokenize(text_enhanced)

        tokens_pre_word_piece_A = dp_split[14].lower()
        tokens_pre_word_piece_B = dp_split[16].lower()

        pronoun = dp_split[7].strip().lower()
        pronoun_index_orig =  int(dp_split[8].strip())
        pronoun_index_orig_enhanced =  int(dp_split[9].strip())

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

        for masked_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
            tokenized_text_A[masked_index] = '<mask>'
        print(tokenized_text_A, "tokenized_text A MASKED")

        for masked_index in range(masked_indices_A_text_enhanced[0], masked_indices_A_text_enhanced[1]):
            tokenized_text_enhanced_A[masked_index] = '<mask>'
        print(tokenized_text_enhanced_A, "tokenized_enchanced_text A MASKED")

        for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
            tokenized_text_B[masked_index] = '<mask>'
        print(tokenized_text_B, "tokenized_text B MASKED")

        for masked_index in range(masked_indices_B_text_enhanced[0], masked_indices_B_text_enhanced[1]):
            tokenized_text_enhanced_B[masked_index] = '<mask>'
        print(tokenized_text_enhanced_B, "tokenized_enchanced_text B MASKED")

        masked_lm_labels_A = []
        masked_lm_labels_B = []

        masked_lm_labels_A_enhanced = []
        masked_lm_labels_B_enhanced = []

        # Convert token to vocabulary indices
        indexed_tokens_A = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_A))
        indexed_tokens_B = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_B))
        indexed_tokens_A_pre_mask = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask))
        indexed_tokens_B_pre_mask = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask))

        #enhanced
        indexed_tokens_A_enhanced = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_A))
        indexed_tokens_B_enhanced = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_enhanced_B))
        indexed_tokens_A_pre_mask_enhanced = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask_enhanced))
        indexed_tokens_B_pre_mask_enhanced = tokenizer.add_special_tokens_single_sentence(tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask_enhanced))

        #mask all labels but wsc options
        for token_index in range(len(indexed_tokens_A)):
            if token_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
                masked_lm_labels_A.append(indexed_tokens_A_pre_mask[token_index])
            else:
                masked_lm_labels_A.append(-1)

        #mask all labels but wsc options
        for token_index in range(len(indexed_tokens_B)):
            if token_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
                masked_lm_labels_B.append(indexed_tokens_B_pre_mask[token_index])
            else:
                masked_lm_labels_B.append(-1)

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


        masked_tokens_A =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A if i!=-1]))
        masked_tokens_B =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B if i!=-1]))

        masked_tokens_A_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A_enhanced if i != -1]))
        masked_tokens_B_enhanced = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B_enhanced if i != -1]))

        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids_A = [0] * len(indexed_tokens_A)
        segments_ids_B = [0] * len(indexed_tokens_B)

        masked_lm_labels_A_non_neg = [(index, item) for index, item in enumerate(masked_lm_labels_A) if item!=-1]
        masked_lm_labels_B_non_neg =  [(index, item) for index, item in enumerate(masked_lm_labels_B) if item!=-1]

        masked_lm_labels_A_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_A_enhanced) if item != -1]
        masked_lm_labels_B_non_neg_enhanced = [(index, item) for index, item in enumerate(masked_lm_labels_B_enhanced) if item != -1]

        # Convert inputs to PyTorch tensors
        tokens_tensor_A = torch.tensor([indexed_tokens_A])
        segments_tensors_A = torch.tensor([segments_ids_A])
        masked_lm_labels_A = torch.tensor([masked_lm_labels_A])

        tokens_tensor_B = torch.tensor([indexed_tokens_B])

        segments_tensors_B = torch.tensor([segments_ids_B])
        masked_lm_labels_B = torch.tensor([masked_lm_labels_B])


        #enhanced
        tokens_tensor_A_enhanced = torch.tensor([indexed_tokens_A_enhanced])
        masked_lm_labels_A_enhanced = torch.tensor([masked_lm_labels_A_enhanced])

        tokens_tensor_B_enhanced = torch.tensor([indexed_tokens_B_enhanced])
        masked_lm_labels_B_enhanced = torch.tensor([masked_lm_labels_B_enhanced])


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

        total_logprobs_A_enhanced = 0
        total_logprobs_B_enhanced = 0

        # mask generation
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

            return perm_mask, target_mapping

        with torch.no_grad():
            perm_mask_A, target_mapping_A = get_masks(tokens_tensor_A, masked_lm_labels_A_non_neg)
            perm_mask_B, target_mapping_B = get_masks(tokens_tensor_B, masked_lm_labels_B_non_neg)
            probs_A = model(tokens_tensor_A, perm_mask=perm_mask_A, target_mapping=target_mapping_A)#, segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
            probs_B = model(tokens_tensor_B, perm_mask=perm_mask_B, target_mapping=target_mapping_B)#, segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

            perm_mask_A_enhanced, target_mapping_A_enhanced = get_masks(tokens_tensor_A_enhanced, masked_lm_labels_A_non_neg_enhanced)
            perm_mask_B_enhanced, target_mapping_B_enhanced = get_masks(tokens_tensor_B_enhanced, masked_lm_labels_B_non_neg_enhanced)
            probs_A_enhanced = model(tokens_tensor_A_enhanced, perm_mask=perm_mask_A_enhanced, target_mapping=target_mapping_A_enhanced)  # , segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
            probs_B_enhanced = model(tokens_tensor_B_enhanced, perm_mask=perm_mask_B_enhanced, target_mapping=target_mapping_B_enhanced)  # , segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

            logprobs_A = torch.nn.functional.log_softmax(probs_A[0], dim=-1)
            logprobs_B = torch.nn.functional.log_softmax(probs_B[0], dim=-1)

            logprobs_A_enhanced = torch.nn.functional.log_softmax(probs_A_enhanced[0], dim=-1)
            logprobs_B_enhanced = torch.nn.functional.log_softmax(probs_B_enhanced[0], dim=-1)

            print("-----------A---------------")

            for n, index_item in enumerate(masked_lm_labels_A_non_neg):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                #print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_A +=  logprobs_A[0,n,item].item()

            for n, index_item in enumerate(masked_lm_labels_A_non_neg_enhanced):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                # print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_A_enhanced += logprobs_A_enhanced[0, n, item].item()

            print("-----------B---------------")

            for n, index_item in enumerate(masked_lm_labels_B_non_neg):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                #print(probs_A[0, index, item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_B += logprobs_B[0,n,item].item()

            for n, index_item in enumerate(masked_lm_labels_B_non_neg_enhanced):
                index, item = index_item
                print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                # print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                total_logprobs_B_enhanced += logprobs_B_enhanced[0, n, item].item()


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