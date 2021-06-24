import re
import numpy as np
import pandas as pd


path_to_wsc = './data/dataset/enhanced_wsc.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')


alpha_re = re.compile(r"[^A-Za-z]+")
def align_word_pieces(spacy_tokens, wp_tokens, retry=True):
    """Align tokens against word-piece toeens. The alignment is returned as a
    list of lists. If alignment[3] == [4, 5, 6], that means that spacy_tokens[3]
    aligns against 3 tokens: wp_tokens[4], wp_tokens[5] and wp_tokens[6].
    All spaCy tokens must align against at least one element of wp_tokens.
    """
    spacy_tokens = list(spacy_tokens)
    wp_tokens = list(wp_tokens)
    if not wp_tokens:
        return [[] for _ in spacy_tokens]
    elif not spacy_tokens:
        return []
    # Check alignment
    spacy_string = "".join(spacy_tokens).lower()
    wp_string = "".join(wp_tokens).lower()
    if not spacy_string and not wp_string:
        return None
    if spacy_string != wp_string:
        if retry:
            # Flag to control whether to apply a fallback strategy when we
            # don't align, of making more aggressive replacements. It's not
            # clear whether this will lead to better or worse results than the
            # ultimate fallback strategy, of calling the sub-tokenizer on the
            # spaCy tokens. Probably trying harder to get alignment is good:
            # the ultimate fallback actually *changes what wordpieces we
            # return*, so we get (potentially) different results out of the
            # transformer. The more aggressive alignment can only change how we
            # map those transformer features to tokens.
            spacy_tokens = [alpha_re.sub("", t) for t in spacy_tokens]
            wp_tokens = [alpha_re.sub("", t) for t in wp_tokens]
            spacy_string = "".join(spacy_tokens).lower()
            wp_string = "".join(wp_tokens).lower()
            if spacy_string == wp_string:
                return _align(spacy_tokens, wp_tokens)
        # If either we're not trying the fallback alignment, or the fallback
        # fails, we return None. This tells the wordpiecer to align by
        # calling the sub-tokenizer on the spaCy tokens.
        return None
    output = _align(spacy_tokens, wp_tokens)
    if len(set(flatten_list(output))) != len(wp_tokens):
        return None
    return output


def _align(seq1, seq2):
    # Map character positions to tokens
    map1 = _get_char_map(seq1)
    map2 = _get_char_map(seq2)
    # For each token in seq1, get the set of tokens in seq2
    # that share at least one character with that token.
    alignment = [set() for _ in seq1]
    unaligned = set(range(len(seq2)))
    for char_position in range(map1.shape[0]):
        i = map1[char_position]
        j = map2[char_position]
        alignment[i].add(j)
        if j in unaligned:
            unaligned.remove(j)
    # Sort, make list
    output = [sorted(list(s)) for s in alignment]
    # Expand alignment to adjacent unaligned tokens of seq2
    for indices in output:
        if indices:
            while indices[0] >= 1 and indices[0] - 1 in unaligned:
                indices.insert(0, indices[0] - 1)
            last = len(seq2) - 1
            while indices[-1] < last and indices[-1] + 1 in unaligned:
                indices.append(indices[-1] + 1)
    return output


def _get_char_map(seq):
    char_map = np.zeros((sum(len(token) for token in seq),), dtype="i")
    offset = 0
    for i, token in enumerate(seq):
        for j in range(len(token)):
            char_map[offset + j] = i
        offset += len(token)
    return char_map


def _tokenize_individual_tokens(model, sent):
    # As a last-chance strategy, run the wordpiece tokenizer on the
    # individual tokens, so that alignment is trivial.
    wp_tokens = []
    sent_align = []
    offset = 0
    for token in sent:
        if token.text.strip():
            subtokens = model.tokenize(token.text)
            wp_tokens.extend(subtokens)
            sent_align.append([i + offset for i in range(len(subtokens))])
            offset += len(subtokens)
        else:
            sent_align.append([])
    return wp_tokens, sent_align

def flatten_list(nested):
    """Flatten a nested list."""
    flat = []
    for x in nested:
        flat.extend(x)
    return flat

"""
def find_sublist(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))
    return results


"""
def find_sublist(sublist, parent):
    results = []
    sub_len = len(sublist)
    valid_starts = [n for n, word in enumerate(parent) if word == sublist[0]]

    for i in valid_starts:
        if parent[i:i + sub_len] == sublist:
            results.append(i)

    return np.array(results)

def get_shorter_lists(l):
    perms = []
    for i in range(len(l)):
        n = l[:]
        n.pop(i)
        perms.append(n)
    return perms


def safe_increment(d, k):
    if k not in d.keys():
        d[k] = 1
    else:
        d[k] += 1


def safe_append(d, k, e):
    if k not in d.keys():
        d[k] = [e]
    else:
        d[k].append(e)


def match_lists(text_subset_list, text_full_list, exp_name):
    # try simple matching
    matches = find_sublist(text_subset_list, text_full_list)
    backoff_strategy = 'none'

    if exp_name in ['text_scrambled']:
        return matches, backoff_strategy


    if len(matches) == 0:
        # back off to shorter list
        backoff_strategy = 'subtract_one'
        shorter_perms = get_shorter_lists(text_subset_list)
        all_matches = []
        for perm in shorter_perms:
            matches = find_sublist(perm, text_full_list)
            if len(matches) != 0:
                all_matches.append(matches)
        # how to select? first is best most of time but if ''s' in sublist, pick last
        if len(all_matches) !=0:
            if "\'" in text_subset_list:
                matches = all_matches[-1]
            else:
                matches = all_matches[0]

        # if first backoff strategy doesn't work
        if len(all_matches) == 0:
            backoff_strategy = 'last_word'
            text_subset_list = [text_subset_list[-1]]
            matches = find_sublist(text_subset_list, text_full_list)

    return matches, backoff_strategy


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
                matches = match_lists(tokens_pre_word_piece_A.split(), text_enhanced.split())
                #print(tokens_pre_word_piece_B)
                #print(text_enhanced)
                #print(matches)

