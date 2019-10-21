import pandas as pd
import re


path_to_wsc = '../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

def find_sublist(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results


def match_lists(text_subset_list, text_full_list):
    #try simple matching
    matches = find_sublist(text_subset_list, text_full_list)
    if len(matches) == 0:
        text_subset_list_backoff = [text_subset_list[-1]]
        matches = find_sublist(text_subset_list_backoff, text_full_list)
        if len(matches) == 0:
            #remove 's
            text_full_list_no_possessive = [token.replace("\'s", "") for token in text_full_list]
            matches = find_sublist(text_subset_list, text_full_list_no_possessive)
            #back off to last word (often the correct one because of 'the')
            if len(matches) == 0:
                text_subset_list_backoff1 = [text_subset_list[-1]]
                matches = find_sublist(text_subset_list_backoff1, text_full_list)
                # back off to word before that
                if len(matches) == 0:
                        text_subset_list_backoff2 = [text_subset_list[-2]]
                        matches = find_sublist(text_subset_list_backoff2, text_full_list)
    return matches


def test_matching():
    for current_alt, current_pron_index in [('text_original', 'pron_index'),
                                            ('text_voice', 'pron_index_voice'),
                                            ('text_tense', 'pron_index_tense'),
                                            ('text_number', 'pron_index_number'),
                                            ('text_gender', 'pron_index'),
                                            ('text_rel_1', 'pron_index_rel'),
                                            ('text_syn', 'pron_index_syn'),
                                            #('text_scrambled', 'pron_index_scrambled'),
                                            ('text_adverb', 'pron_index_adverb')
                                            ]:

        for q_index, dp_split in wsc_datapoints.iterrows():
            if dp_split[current_alt].replace(' ', '') != '-' and dp_split[current_alt].replace(' ', ''):

                text_enhanced = "[CLS] " + dp_split[current_alt] + " [SEP]"
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

                #get ref. matches
                tokens_pre_word_piece_A = re.sub(' +', ' ', tokens_pre_word_piece_A.rstrip().lower())
                text_enhanced = re.sub(' +', ' ', text_enhanced.lower())
                match_lists(tokens_pre_word_piece_A.split(), text_enhanced.split())

