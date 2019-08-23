import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from copy import deepcopy

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../data/wsc_data/enhanced.dset.tsv'
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
    # [tokenized_text.insert(pronoun_index, i) for i in tokenized_option]
    new_pronoun_index = pronoun_index + len(tokenized_option)
    tokenized_text.pop(new_pronoun_index)
    return tokenized_text

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

correct_preds = 0
correct_preds_original = 0
stability_match  = 0

all_preds = 0

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

for dp in wsc_datapoints[1:]:
    dp_split = dp.split('\t')
    if dp_split[1].replace(' ', '') != '-':
        # Tokenized input
        correct_answer = dp_split[-3]
        #check for empty

        #text_A = "[CLS] " + dp_split[3] + "[SEP]"
        #text_B = "[CLS] " + dp_split[6]  + "[SEP]"
        text = "[CLS] " + dp_split[0]  + " [SEP]"
        text = text.lower()
        #text_original = "[CLS] " + dp_split[1]  + " [SEP]"
        #print(text, "text")

        #text_A_split = text_A.split()
        #text_B_split = text_B.split()

        #tokenized_text_A = tokenizer.tokenize(text_A)
        #tokenized_text_B = tokenizer.tokenize(text_B)
        #tokenized_original_text = tokenizer.tokenize(text_original)

        tokens_pre_word_piece_A_orig = dp_split[5]
        tokens_pre_word_piece_B_orig = dp_split[7]

        tokens_pre_word_piece_A = dp_split[6]
        tokens_pre_word_piece_B = dp_split[8]

        pronoun = dp_split[2].strip()
        pronoun_index_orig =  int(dp_split[3].strip())
        #pronoun_index_orig_original =  int(dp_split[4].strip())

        tokenized_option_A = tokenizer.tokenize(tokens_pre_word_piece_A)
        tokenized_option_B = tokenizer.tokenize(tokens_pre_word_piece_B)

        tokenized_option_A_orig = tokens_pre_word_piece_A_orig.split()
        tokenized_option_B_orig = tokens_pre_word_piece_B_orig.split()
        
        tokenized_pronoun = tokenizer.tokenize(pronoun)

        tokenized_option_A_len = len(tokenized_option_A)
        tokenized_option_B_len = len(tokenized_option_B)

        if tokenized_option_A_len == 1 and tokenized_option_B_len == 1:

            #replace each occurance of orig. referents with their synonyms
            original_text = deepcopy(text)

            text = text.replace(tokens_pre_word_piece_A_orig, tokens_pre_word_piece_A)
            text = text.replace(tokens_pre_word_piece_B_orig, tokens_pre_word_piece_B)

            tokenized_text = tokenizer.tokenize(text)

            #tokenize properly
            tokenized_option_A_orig = tokenizer.tokenize(tokens_pre_word_piece_A_orig)
            tokenized_option_B_orig = tokenizer.tokenize(tokens_pre_word_piece_B_orig)

            tokenized_original_text = tokenizer.tokenize(original_text)

            print(tokenized_option_A, "tokenized_option A after")
            print(tokenized_option_B, "tokenized_option B after")

            print(original_text, "original_text")
            print(text, "text after")

            matched_pronouns_text = find_sub_list(tokenized_pronoun, tokenized_text)
            matched_pronouns_original_text = find_sub_list(tokenized_pronoun,  tokenized_original_text)

            first_indices_text = np.array([mp[0] for mp in matched_pronouns_text])
            first_indices_text_original = np.array([mp[0] for mp in matched_pronouns_original_text])

            print(matched_pronouns_text, "matched_pronouns_text")
            print(matched_pronouns_original_text, "matched_pronouns_text_original")

            correct_idx_text = (np.abs(first_indices_text - pronoun_index_orig)).argmin()
            correct_idx_text_original =  correct_idx_text
            #print(correct_idx_text_original, " correct_idx_text_original")

            pronoun_index_text = matched_pronouns_text[correct_idx_text][0]
            pronoun_index_text_original  = matched_pronouns_original_text[correct_idx_text_original][0]


            tokenized_text_A = replace_pronoun(tokenized_text, pronoun_index_text, tokenized_option_A)
            tokenized_text_B = replace_pronoun(tokenized_text, pronoun_index_text, tokenized_option_B)

            tokenized_text_original_A = replace_pronoun(tokenized_original_text, pronoun_index_text_original, tokenized_option_A_orig)
            tokenized_text_original_B = replace_pronoun(tokenized_original_text, pronoun_index_text_original, tokenized_option_B_orig)

            print(tokenized_text_A, "tokenized_text_A")
            print(tokenized_text_original_A, "tokenized_text_original_A")

            matched_A_text = find_sub_list(tokenized_option_A, tokenized_text_A)
            matched_B_text = find_sub_list(tokenized_option_B, tokenized_text_B)

            matched_A_text_original = find_sub_list(tokenized_option_A_orig, tokenized_text_original_A)
            matched_B_text_original = find_sub_list(tokenized_option_B_orig, tokenized_text_original_B)

            print(matched_A_text, "matched A")
            print(matched_A_text_original, "matched A original")

            masked_indices_A_text = [m for m in matched_A_text if m[0] == pronoun_index_text][0]
            masked_indices_A_text_original = [m for m in matched_A_text_original if m[0] == pronoun_index_text_original][0]

            masked_indices_B_text = [m for m in matched_B_text if m[0] == pronoun_index_text][0]
            masked_indices_B_text_original = [m for m in matched_B_text_original if m[0] == pronoun_index_text_original][0]


            tokenized_text_A_pre_mask = deepcopy(tokenized_text_A)
            tokenized_text_B_pre_mask = deepcopy(tokenized_text_B)

            tokenized_text_A_pre_mask_original = deepcopy(tokenized_text_original_A)
            tokenized_text_B_pre_mask_original = deepcopy(tokenized_text_original_B)

            for masked_index in range(masked_indices_A_text[0], masked_indices_A_text[1]):
                tokenized_text_A[masked_index] = '[MASK]'
            print(tokenized_text_A, "tokenized_text A MASKED")

            for masked_index in range(masked_indices_A_text_original[0], masked_indices_A_text_original[1]):
                tokenized_text_original_A[masked_index] = '[MASK]'
            print(tokenized_text_original_A, "tokenized_enchanced_text A MASKED")

            for masked_index in range(masked_indices_B_text[0], masked_indices_B_text[1]):
                tokenized_text_B[masked_index] = '[MASK]'
            print(tokenized_text_B, "tokenized_text B MASKED")

            for masked_index in range(masked_indices_B_text_original[0], masked_indices_B_text_original[1]):
                tokenized_text_original_B[masked_index] = '[MASK]'
            print(tokenized_text_original_B, "tokenized_enchanced_text B MASKED")

            masked_lm_labels_A = []
            masked_lm_labels_B = []

            masked_lm_labels_A_original = []
            masked_lm_labels_B_original = []

            # Convert token to vocabulary indices
            indexed_tokens_A = tokenizer.convert_tokens_to_ids(tokenized_text_A)
            indexed_tokens_B = tokenizer.convert_tokens_to_ids(tokenized_text_B)
            indexed_tokens_A_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask)
            indexed_tokens_B_pre_mask = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask)

            #original
            indexed_tokens_A_original = tokenizer.convert_tokens_to_ids(tokenized_text_original_A)
            indexed_tokens_B_original = tokenizer.convert_tokens_to_ids(tokenized_text_original_B)
            indexed_tokens_A_pre_mask_original = tokenizer.convert_tokens_to_ids(tokenized_text_A_pre_mask_original)
            indexed_tokens_B_pre_mask_original = tokenizer.convert_tokens_to_ids(tokenized_text_B_pre_mask_original)

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

            #mask all labels but wsc options (original)
            for token_index in range(len(indexed_tokens_A_original)):
                if token_index in range(masked_indices_A_text_original[0], masked_indices_A_text_original[1]):
                    masked_lm_labels_A_original.append(indexed_tokens_A_pre_mask_original[token_index])
                else:
                    masked_lm_labels_A_original.append(-1)

            #mask all labels but wsc options
            for token_index in range(len(indexed_tokens_B_original)):
                if token_index in range(masked_indices_B_text_original[0], masked_indices_B_text_original[1]):
                    masked_lm_labels_B_original.append(indexed_tokens_B_pre_mask_original[token_index])
                else:
                    masked_lm_labels_B_original.append(-1)


            masked_tokens_A =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A if i!=-1]))
            masked_tokens_B =  ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B if i!=-1]))

            masked_tokens_A_original = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_A_original if i != -1]))
            masked_tokens_B_original = ' '.join(tokenizer.convert_ids_to_tokens([i for i in masked_lm_labels_B_original if i != -1]))

            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids_A = [0] * len(indexed_tokens_A)
            segments_ids_B = [0] * len(indexed_tokens_B)

            masked_lm_labels_A_non_neg = [(index, item) for index, item in enumerate(masked_lm_labels_A) if item!=-1]
            masked_lm_labels_B_non_neg =  [(index, item) for index, item in enumerate(masked_lm_labels_B) if item!=-1]

            masked_lm_labels_A_non_neg_original = [(index, item) for index, item in enumerate(masked_lm_labels_A_original) if item != -1]
            masked_lm_labels_B_non_neg_original = [(index, item) for index, item in enumerate(masked_lm_labels_B_original) if item != -1]

            # Convert inputs to PyTorch tensors
            tokens_tensor_A = torch.tensor([indexed_tokens_A])
            segments_tensors_A = torch.tensor([segments_ids_A])
            masked_lm_labels_A = torch.tensor([masked_lm_labels_A])

            tokens_tensor_B = torch.tensor([indexed_tokens_B])

            segments_tensors_B = torch.tensor([segments_ids_B])
            masked_lm_labels_B = torch.tensor([masked_lm_labels_B])


            #original
            tokens_tensor_A_original = torch.tensor([indexed_tokens_A_original])
            masked_lm_labels_A_original = torch.tensor([masked_lm_labels_A_original])

            tokens_tensor_B_original = torch.tensor([indexed_tokens_B_original])
            masked_lm_labels_B_original = torch.tensor([masked_lm_labels_B_original])


            # If you have a GPU, put everything on cuda
            tokens_tensor_A = tokens_tensor_A.to(device=device)
            tokens_tensor_B = tokens_tensor_B.to(device=device)
            segments_tensors_A = segments_tensors_A.to(device=device)
            segments_tensors_B = segments_tensors_B.to(device=device)
            masked_lm_labels_A =  masked_lm_labels_A.to(device=device)
            masked_lm_labels_B =  masked_lm_labels_B.to(device=device)
            print(masked_lm_labels_A, " masked_lm_labels_A tensor")

            model.to(device=device)

            # Predict all tokens
            total_logprobs_A = 0
            total_logprobs_B = 0

            total_logprobs_A_original = 0
            total_logprobs_B_original = 0

            with torch.no_grad():
                probs_A = model(tokens_tensor_A)#, segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
                probs_B = model(tokens_tensor_B)#, segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

                probs_A_original = model(tokens_tensor_A_original)  # , segments_tensors_A) #, masked_lm_labels =  masked_lm_labels_A)
                probs_B_original = model(tokens_tensor_B_original)  # , segments_tensors_B) #, masked_lm_labels =  masked_lm_labels_B)

                logprobs_A = torch.nn.functional.log_softmax(probs_A, dim=-1)
                logprobs_B = torch.nn.functional.log_softmax(probs_B, dim=-1)

                logprobs_A_original = torch.nn.functional.log_softmax(probs_A_original, dim=-1)
                logprobs_B_original = torch.nn.functional.log_softmax(probs_B_original, dim=-1)

                print("-----------A---------------")

                for index_item in masked_lm_labels_A_non_neg:
                    index, item = index_item
                    print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                    #print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                    total_logprobs_A +=  logprobs_A[0,index,item].item()

                for index_item in masked_lm_labels_A_non_neg_original:
                    index, item = index_item
                    print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                    print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                    total_logprobs_A_original += logprobs_A_original[0, index, item].item()

                print("-----------B---------------")

                for index_item in masked_lm_labels_B_non_neg:
                    index, item = index_item
                    print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                    #print(probs_A[0, index, item].item(), " : probs_A[0,index,item].item()")
                    total_logprobs_B += logprobs_B[0,index,item].item()

                for index_item in masked_lm_labels_B_non_neg_original:
                    index, item = index_item
                    print(index, tokenizer.convert_ids_to_tokens([item]), " : index, item")
                    print(probs_A[0,index,item].item(), " : probs_A[0,index,item].item()")
                    total_logprobs_B_original += logprobs_B_original[0, index, item].item()


                print(total_logprobs_A / tokenized_option_A_len  , " total_probs_A / tokenized_option_A_len")
                print(total_logprobs_B / tokenized_option_B_len , " total_probs_B / tokenized_option_B_len")
                print(correct_answer.strip().strip('.').replace(' ', ''), " correct_answer")

                print(total_logprobs_A_original / tokenized_option_A_len, " total_probs_A / tokenized_option_A_len")
                print(total_logprobs_B_original / tokenized_option_B_len, " total_probs_B / tokenized_option_B_len")
                print(correct_answer.strip().strip('.').replace(' ', ''), " correct_answer")

                max_index = np.argmax([total_logprobs_A / tokenized_option_A_len, total_logprobs_B / tokenized_option_B_len ])
                max_index_original = np.argmax([total_logprobs_A_original / tokenized_option_A_len, total_logprobs_B_original
                                                / tokenized_option_B_len ])

                prediction = "A" if max_index == 0 else "B"
                prediction_original = "A" if max_index_original == 0 else "B"

                print(prediction, " prediction")
                print(prediction_original, " prediction original")

                if prediction == correct_answer.strip().strip('.').replace(' ', ''):
                    correct_preds += 1
                if prediction_original == correct_answer .strip().strip('.').replace(' ', ''):
                    correct_preds_original += 1
                if prediction_original == prediction:
                    stability_match += 1

                all_preds += 1
                print("#############################################################################")
        else:
            continue
    else:
        continue

accuracy = correct_preds/all_preds
print(all_preds, " : all_preds")
print(correct_preds, " : correct_preds")
print(accuracy, " : accuracy")

accuracy_original = correct_preds_original/all_preds
print(all_preds, " : all_preds")
print(correct_preds_original, " : correct_preds original")
print(accuracy_original, " : accuracy_originaly")

print(stability_match, ": stability_match")
print(stability_match / all_preds , ": stability_match %")