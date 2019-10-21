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


# Load pre-trained model tokenizer (vocabulary)
vocab = []

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


      for q_index, dp_split in wsc_datapoints.iterrows():
        if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):
            # save the index
            # Tokenized input
            correct_answer = dp_split['correct_answer'].strip().strip('.').replace(' ', '')
            text_enhanced = dp_split[current_alt]

            tokenized_enhanced_text = [t.strip().strip('\n').replace(' ', '') for t in text_enhanced.split()]
            vocab.extend([t for t in tokenized_enhanced_text])

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


            tokenized_tokens_pre_word_piece_A = [t.strip().strip('\n').replace(' ', '').strip() for t in tokens_pre_word_piece_A.split()]
            tokenized_tokens_pre_word_piece_B = [t.strip().strip('\n').replace(' ', '').strip() for t in tokens_pre_word_piece_B.split()]

            vocab.extend([t for t in tokenized_tokens_pre_word_piece_A + tokenized_tokens_pre_word_piece_B])

            if current_alt == 'text_gender':
                pronoun = dp_split['pron_gender'].strip()
            elif current_alt == 'text_number':
                pronoun = dp_split['pron_number'].strip()
            else:
                pronoun = dp_split['pron'].strip()

            vocab.append(pronoun)

vocab = list(set(vocab))

print(vocab)

#print(description)
with open('full_vocab_all_perturbs.pickle', 'wb') as f:
    pickle.dump(vocab, f)
