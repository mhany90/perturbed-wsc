import sys
import pickle
import random

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from collections import Counter
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from helpers import match_lists, find_sublist, safe_append, safe_increment

TSV_PATH = '../data/final.tsv'
EXPERIMENT_ARR = [('text_original', 'pron_index'),
                  ('text_voice', 'pron_index_voice'),
                  ('text_tense', 'pron_index_tense'),
                  ('text_number', 'pron_index_number'),
                  ('text_gender', 'pron_index'),
                  ('text_rel_1', 'pron_index_rel'),
                  ('text_syn', 'pron_index_syn'),
                  ('text_adverb', 'pron_index_adverb')]

# initialise
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
datafile = pd.read_csv(TSV_PATH, sep='\t')
model_name = 'bert-base-uncased' if '--debug' in sys.argv else 'bert-large-uncased'
NUM_HEADS = 12 if '--debug' in sys.argv else 16

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True, output_attentions=True).eval().to(device)

for exp_name, pron_col in EXPERIMENT_ARR:
    print(exp_name)
    representations = []

    for q_index, entry in tqdm.tqdm(datafile.iterrows()):
        if entry[exp_name].replace(' ', '') in [None, '-']:
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
        matched_referents_A, backoff_strategy_A = match_lists(tokens_option_A, tokens_orig, exp_name)
        matched_referents_B, backoff_strategy_B = match_lists(tokens_option_B, tokens_orig, exp_name)

        if len(matched_referents_A) == 0 or len(matched_referents_B) == 0:
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

        if discrim_indices and referent_indices_A and referent_indices_B:
            with torch.no_grad():
                _, sent_hidden, _ = model(ids_orig.unsqueeze(0))

            sent_hidden = sent_hidden[-1].squeeze()
            reps_A = sent_hidden[referent_indices_A]
            reps_B = sent_hidden[referent_indices_B]
            reps_correct = sent_hidden[referent_indices_A] if correct_answer == "A" else sent_hidden[referent_indices_B]
            reps_wrong = sent_hidden[referent_indices_A] if correct_answer == "B" else sent_hidden[referent_indices_B]
            reps_discrim = sent_hidden[discrim_indices]

            reps_A = reps_A.mean(dim=0)
            reps_B = reps_B.mean(dim=0)
            reps_correct = reps_correct.mean(dim=0)
            reps_wrong = reps_wrong.mean(dim=0)
            reps_discrim = reps_discrim.mean(dim=0)

            representations.append((torch.cat([reps_A, reps_B, reps_discrim], dim=-1).cpu(),
                                    ["A", "B"].index(correct_answer)))

    X = torch.stack([i[0] for i in representations])
    y = torch.tensor([i[1] for i in representations], dtype=torch.float)
    indices = torch.randperm(X.size(0))
    X[indices], y[indices] = X[indices], y[indices]

    results = []
    kf = KFold(n_splits=10)
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        clf = nn.Sequential(
            nn.Linear(3 * 768, 1)
        )

        transform = nn.Sequential(nn.Linear(768, 768))

        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CosineEmbeddingLoss()
        optimizer = torch.optim.SGD(clf.parameters(), lr=0.01)

        clf.train()
        for epoch in range(10):
            X_A = transform(X_train[:, :768])
            X_B = transform(X_train[:, 768:(768 * 2)])
            X_discrim = X_train[:, (768 * 2):(768 * 3)]
            pred = clf(X_train).squeeze()

            optimizer.zero_grad()
            l1 = criterion(X_A, X_discrim, y_train)
            l1.backward()
            optimizer.step()

            optimizer.zero_grad()
            l2 = criterion(X_B, X_discrim, (1 - y_train))
            l2.backward()
            optimizer.step()

        clf.eval()
        X_A = transform(X_test[:, :768])
        X_B = transform(X_test[:, 768:(768 * 2)])
        X_discrim = X_test[:, (768 * 2):(768 * 3)]
        sim_A = F.cosine_similarity(X_A, X_discrim)
        sim_B = F.cosine_similarity(X_B, X_discrim)
        correct = ((sim_B > sim_A).type(torch.float) == y_test).nonzero().size(0)
        total = y_test.size(0)

        # pred = torch.sigmoid(clf(X_test).squeeze())
        # pred[pred > 0.5] = 1
        # pred[pred <= 0.5] = 0
        #
        # correct = (pred == y_test).nonzero().size(0)
        # total = pred.size(0)
        results.append(correct / total)

    print("acc: {}".format(np.mean(results)))

