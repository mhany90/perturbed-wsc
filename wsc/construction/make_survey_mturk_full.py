import torch
import numpy as np
import sys
import pandas as pd
import csv


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

path_to_wsc = '../../data/wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.adverb.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')

#wsc_file = open(path_to_wsc, 'r')
#wsc_datapoints = wsc_file.readlines()

def insert_highlight(text_tokenized, pronoun_index):
    text_tokenized.insert(pronoun_index, "<font color='red'> <b>")
    text_tokenized.insert(pronoun_index + 2, "</b> </font>")

    return text_tokenized

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e.lower() == sl[0].lower()):
        item = str(l[ind:ind+sll]).lower()
        sub_item = str(sl).lower()
        if  item == sub_item:
            results.append((ind,ind+sll))
    return results

def replace_pronoun(tokenized_text, pronoun_index, tokenized_option):
    tokenized_text = tokenized_text[:pronoun_index] + tokenized_option + tokenized_text[pronoun_index:]
    new_pronoun_index = pronoun_index + len(tokenized_option)
    tokenized_text.pop(new_pronoun_index)
    return tokenized_text

counter = 0
limit_of_dps = 289
split_counter = 0
filename = sys.argv[1]
filename = filename + '.full' + '.csv'
header = ['wsc_id', 'text','answer_a','answer_b']
lines = []
lines.append(header)

for q_index, dp_split in wsc_datapoints.iterrows():
        #if counter % limit_of_dps == 0 or counter == 0:
            #split_counter = split_counter + 1
            #fsplitname = str(filename) + "." + str(split_counter) + '.csv'
            #if counter != 0:
            #    f.close()
            #f = open(fsplitname, 'w')
            #f.write(header)

        correct_answer = dp_split['correct_answer'].strip()
        pronoun = dp_split['pron'].strip()
        pronoun_index_orig =  dp_split['pron_index']

        answer_A = dp_split['answer_a'].strip()
        answer_B = dp_split['answer_b'].strip()
        text = dp_split['text_scrambled'].strip()
        """
        print(text, ' : text')
        print(answer_A, " : answer_A")
        print(answer_B, " : answer_B")
        print(pronoun, " : pronoun")
        print(pronoun_index_orig, " : pronoun_index_orig")
        """

        if  text != '-' and  text:
            counter = counter + 1
            text_tokenized = text.split()
            pronoun_index_orig = int(pronoun_index_orig)

            #match pron
            matched_pronouns = find_sub_list([pronoun], text_tokenized)
            first_indices = np.array([mp[0] for mp in  matched_pronouns])
            correct_idx = (np.abs(first_indices - pronoun_index_orig)).argmin()
            pronoun_index_text = matched_pronouns[correct_idx][0]

            text_tokenized_highlighted = insert_highlight(text_tokenized, pronoun_index_text)

            #label = "l: " + str(counter)
            #type = "t: radio"
            id = str(q_index)
            question = ' '.join(text_tokenized_highlighted) + " ".replace('\n', '') #+ '"'
            answera = "<b> A) </b> " + answer_A.replace('\n', '')
            answerb = "<b> B) </b> " + answer_B.replace('\n', '')
            lines.append([id, question, answera, answerb])


csv.register_dialect('myDialect',
quoting=csv.QUOTE_ALL,
skipinitialspace=True)


with open(filename, 'w') as writeFile:
    writer = csv.writer(writeFile, dialect='myDialect')
    writer.writerows(lines)

writeFile.close()


    #f.write(question)
            #f.write(',')
            #f.write(answera)
            #f.write(',')
            #f.write(answerb)
            #f.write('\n')


            #print(label)
            #print(type)
            #print(question)
            #print(answera)
            #print(answerb)
            #print('\n')












