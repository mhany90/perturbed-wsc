import sys
import pickle

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from collections import Counter
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from helpers import match_lists, find_sublist


TSV_PATH = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
EXPERIMENT_ARR = [('text_original', 'pron_index'),
                  ('text_voice', 'pron_index_voice'),
                  ('text_tense', 'pron_index_tense'),
                  ('text_context', 'pron_index_context'),
                  ('text_number', 'pron_index_number'),
                  ('text_gender', 'pron_index'),
                  ('text_rel_1', 'pron_index_rel'),
                  ('text_syn', 'pron_index_syn'),
                  ('text_scrambled', 'pron_index_scrambled'),
                  ('text_freqnoun', 'pron_index_freqnoun'),
                  ('text_adverb', 'pron_index_adverb')]

# save to these
original_predictions = []
accuracies = {}
stabilities = {}
answers = {}
attentions = {}
tuples = {}
indices = {}
closer_referents = {}

# initialise
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
datafile = pd.read_csv(TSV_PATH, sep='\t')
model_name = 'bert-base-uncased' if sys.argv[1] == '--debug' else 'bert-large-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True).eval().to(device)

for exp_name, pron_col in EXPERIMENT_ARR:
    print(exp_name)

    # initialise output stuff
    accuracies[exp_name] = {'all': 0}
    stabilities[exp_name] = {'all': 0}
    # should be full-sized for every perturbation
    answers[exp_name] = {'gold': [], 'pred': [], 'probs': []}
    attentions[exp_name] = {'correct': {'pred': [], 'gold': [], 'cos_d': [], 'cos_o': []},
                            'wrong': {'pred': [], 'gold': [], 'cos_d': [], 'cos_o': []},
                            'discrim': {'': []}}

    tuples[exp_name] = {'correct': {'pred_top1': [], 'gold_top1': [], 'pred_top5': [], 'gold_top5': []},
                        'wrong': {'pred_top1': [], 'gold_top1': [], 'pred_top5': [], 'gold_top5': []},
                        'discrim': {'_top1': [], '_top5': []}}

    closer_referents[exp_name] = {'all':0,'correct':0, 'incorrect':0}

    # which examples have been gathered for each analysis type
    indices[exp_name] = {'attn': [], 'tuples': []}

    total = 0

    for q_index, entry in tqdm.tqdm(datafile.iterrows()):
        if entry[exp_name].replace(' ', '') in [None, '-']:
            answers[exp_name]['gold'].append(-1)
            answers[exp_name]['pred'].append(-1)
            answers[exp_name]['probs'].append((0, 0))
            continue

        correct_answer = entry['correct_answer'].strip().strip('.').replace(' ', '')

        text_orig = "[CLS] " + entry[exp_name] + " [SEP]"
        tokens_orig = tokenizer.tokenize(text_orig)
        ids_orig = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_orig)).to(device)

        # altered answers
        suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_syn', 'text_gender', 'text_number'] else ""
        text_option_A = entry['answer_a{}'.format(suffix)]
        text_option_B = entry['answer_b{}'.format(suffix)]

        tokens_option_A = tokenizer.tokenize(text_option_A)
        tokens_option_B = tokenizer.tokenize(text_option_B)

        suffix = "_" + exp_name.split("_")[1] if exp_name in ['text_gender', 'text_number'] else ""
        text_pron = entry['pron{}'.format(suffix)]
        tokens_pron = tokenizer.tokenize(text_pron)

        # fix pronoun index
        pron_index = int(entry[pron_col])
        matched_prons = find_sublist(tokens_pron, tokens_orig)
        best_match = (np.abs(matched_prons - pron_index)).argmin()
        pron_index = matched_prons[best_match]

        # find referent index(es)
        ignore_attention = False
        referent_indices_A, referent_indices_B = [], []
        if exp_name is not 'text_scrambled':
            matched_referents_A, backoff_strategy_A = match_lists(tokens_option_A, tokens_orig)
            matched_referents_B, backoff_strategy_B = match_lists(tokens_option_B, tokens_orig)

        else:
            matched_referents_A = find_sublist(tokens_option_A, tokens_orig)
            matched_referents_B = find_sublist(tokens_option_B, tokens_orig)
            backoff_strategy_A = 'none'
            backoff_strategy_B = 'none'


        if len(matched_referents_A) == 0 or len(matched_referents_B) == 0:
            print("NO MATCH")
            ignore_attention = True
        else:
            if backoff_strategy_A == 'none':
                referent_indices_A = range(matched_referents_A[0], matched_referents_A[0] + len(tokens_option_A))
            elif backoff_strategy_A == 'subtract_one':
                referent_indices_A = range(matched_referents_A[0], matched_referents_A[0] + len(tokens_option_A) - 1)
            elif backoff_strategy_A == 'last_word':
                referent_indices_A = range(matched_referents_A[0], matched_referents_A[0] + 1)

            if backoff_strategy_B == 'none':
                referent_indices_B = range(matched_referents_B[0], matched_referents_B[0] + len(tokens_option_B))
            elif backoff_strategy_B == 'subtract_one':
                referent_indices_B = range(matched_referents_B[0], matched_referents_B[0] + len(tokens_option_B) - 1)
            elif backoff_strategy_B == 'last_word':
                referent_indices_B = range(matched_referents_B[0], matched_referents_B[0] + 1)

            # find referent that is closer to pron_index
            if abs(matched_referents_A[0] - pron_index) > abs(matched_referents_B[0] - pron_index):
                closer_referent = 'B'
            else:
                closer_referent = 'A'

        # find discrim word
        text_discrim = entry['discrim_word']
        ignore_discrim = False
        discrim_indices = []
        if not isinstance(text_discrim, str):
            ignore_discrim = True
        else:
            text_discrim = text_discrim.strip()
            tokens_discrim = tokenizer.tokenize(text_discrim)
            ids_discrim = tokenizer.convert_tokens_to_ids(tokens_discrim)
            matched_discrim = find_sublist(tokens_discrim, tokens_orig)
            if len(matched_discrim) == 0:
                ignore_discrim = True
            else:
                discrim_indices = range(matched_discrim[0], matched_discrim[0] + len(tokens_discrim))

        # text with referents instead of pronouns
        tokens_A = tokens_orig[:pron_index] + tokens_option_A + tokens_orig[pron_index+1:]
        tokens_B = tokens_orig[:pron_index] + tokens_option_B + tokens_orig[pron_index+1:]
        ids_A = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_A)).to(device)
        ids_B = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_B)).to(device)

        # text with [MASK]s instead of pronouns
        tokens_masked_A = tokens_A[:pron_index] + ['[MASK]'] * len(tokens_option_A) + tokens_orig[pron_index+1:]
        tokens_masked_B = tokens_B[:pron_index] + ['[MASK]'] * len(tokens_option_B) + tokens_orig[pron_index+1:]
        ids_masked_A = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_masked_A)).to(device)
        ids_masked_B = torch.tensor(tokenizer.convert_tokens_to_ids(tokens_masked_B)).to(device)

        predict_indices_A = (ids_masked_A != ids_A).nonzero(as_tuple=True)[0].tolist()
        predict_indices_B = (ids_masked_B != ids_B).nonzero(as_tuple=True)[0].tolist()
        predict_items_A = ids_A[predict_indices_A]
        predict_items_B = ids_B[predict_indices_B]

        with torch.no_grad():
            # pad with batch dim
            probs_A = model(ids_masked_A.unsqueeze(0))[0]
            probs_B = model(ids_masked_B.unsqueeze(0))[0]

            logprobs_A = F.log_softmax(probs_A, dim=-1)
            logprobs_B = F.log_softmax(probs_B, dim=-1)

            total_logprobs_A = sum([logprobs_A[0, index, item].item()
                                    for index, item in zip(predict_indices_A, predict_items_A)]) / len(tokens_option_A)
            total_logprobs_B = sum([logprobs_B[0, index, item].item()
                                   for index, item in zip(predict_indices_B, predict_items_B)]) / len(tokens_option_B)

        predicted_answer = np.argmax([total_logprobs_A, total_logprobs_B])
        answers[exp_name]['pred'].append(predicted_answer)
        answers[exp_name]['gold'].append(0 if correct_answer == "A" else 1)
        predicted_answer = "A" if predicted_answer == 0 else "B"

        if closer_referent == predicted_answer:
            closer_referents[exp_name]['all'] += 1

        correct_logprobs, wrong_logprobs = total_logprobs_A, total_logprobs_B
        if correct_answer == "B":
            correct_logprobs, wrong_logprobs = wrong_logprobs, correct_logprobs
        answers[exp_name]['probs'].append((correct_logprobs, wrong_logprobs))

        if exp_name == 'text_original':
            original_predictions.append(predicted_answer)

        if predicted_answer == correct_answer:
            accuracies[exp_name]['all'] += 1
            if closer_referent == predicted_answer:
                closer_referents[exp_name]['correct'] += 1
            else:
                closer_referents[exp_name]['incorrect'] += 1


        if predicted_answer == original_predictions[q_index]:
            stabilities[exp_name]['all'] += 1

        if not ignore_attention:
            with torch.no_grad():
                rep_orig, attn_orig = model(ids_orig.unsqueeze(0))[1:]

            gold_referent = referent_indices_A if correct_answer == "A" else referent_indices_B
            gold_referent_x = referent_indices_A if correct_answer == "B" else referent_indices_B

            pred_referent = referent_indices_A if predicted_answer == "A" else referent_indices_B
            pred_referent_x = referent_indices_A if predicted_answer == "B" else referent_indices_B

            discrim_referent = discrim_indices

            def fill_attention(ref, name):
                # get layer-head matrix and save
                lhm = torch.stack(attn_orig).squeeze(1)[:, :, pron_index, ref].mean(dim=-1)
                # check for nan
                if torch.isnan(lhm).any().cpu() and name != ('discrim', ''):
                    print(lhm, " lhm")
                    print(ref, " : ref")
                    print(referent_indices_B, "referent_indices_B")
                    print(name, " : name")
                    print(correct_answer, " correct_answer")
                    print(text_orig, "text_orig")
                    print(tokens_A, "tokens_a")
                    print(matched_referents_A, "matched_referents_A")
                    print(backoff_strategy_A, "backoff_strategy_A")
                    print(tokens_B, "tokens_b")
                    print(matched_referents_B, "matched_referents_B")
                    print(backoff_strategy_B, "backoff_strategy_B")
                    exit()

                attentions[exp_name][name[0]][name[1]].append(lhm.cpu())
                num_layers, num_heads = lhm.shape

                top1 = lhm.argmax().item()
                top5 = lhm.view(-1).topk(5).indices.tolist()

                tuples[exp_name][name[0]][name[1] + "_top1"].append((top1 // num_heads, top1 % num_heads))
                for n in top5:
                    tuples[exp_name][name[0]][name[1] + "_top5"].append((n // num_heads, n % num_heads))

            fill_attention(gold_referent, ('correct', 'gold'))
            fill_attention(pred_referent, ('correct', 'pred'))
            fill_attention(gold_referent_x, ('wrong', 'gold'))
            fill_attention(pred_referent_x, ('wrong', 'pred'))
            fill_attention(discrim_referent, ('discrim', ''))
            indices[exp_name]['tuples'].append(q_index)

            # cosine sim (only for gold)
            current_rep = torch.stack(rep_orig).squeeze()
            correct_rep = current_rep[:, gold_referent].mean(dim=1).squeeze()
            wrong_rep = current_rep[:, gold_referent_x].mean(dim=1).squeeze()
            discrim_rep = current_rep[:, discrim_referent].mean(dim=1).squeeze()

            other_indices = list(set(range(len(tokens_orig))) - \
                            (set(gold_referent) | set(gold_referent_x) | set(discrim_referent)))

            # calculate mean after sim!
            others = current_rep[:, other_indices].squeeze()
            attentions[exp_name]['correct']['cos_d'].append(F.cosine_similarity(correct_rep, discrim_rep, dim=1).cpu())
            attentions[exp_name]['wrong']['cos_d'].append(F.cosine_similarity(wrong_rep, discrim_rep, dim=1).cpu())

            attentions[exp_name]['correct']['cos_o'].append(torch.stack([F.cosine_similarity(correct_rep, i, dim=1) for
                                                                         i in others.transpose(0, 1)]).mean().cpu())
            attentions[exp_name]['wrong']['cos_o'].append(torch.stack([F.cosine_similarity(wrong_rep, i, dim=1) for
                                                                       i in others.transpose(0, 1)]).mean().cpu())


        total += 1

    print("accuracy: {}/{} = {}".format(accuracies[exp_name]['all'], total,
                                        accuracies[exp_name]['all'] / total))
    print("stability: {}/{} = {}".format(stabilities[exp_name]['all'], total,
                                         stabilities[exp_name]['all'] / total))
    print("count closer referent: {}".format(closer_referents[exp_name]['all']))

    # for truth in attentions[exp_name]:
    #     for type in attentions[exp_name][truth]:
    #         attentions[exp_name][truth][type] = torch.stack(attentions[exp_name][truth][type]).mean(dim=0).cpu().numpy()

with open('bert.dump', 'wb') as f:
    pickle.dump((answers, indices, tuples, attentions), f)
