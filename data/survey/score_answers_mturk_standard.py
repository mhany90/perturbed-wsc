import pandas as pd
import sys
import numpy as np
import re

def most_frequent(List):
    return max(set(List), key = List.count)

path_to_wsc = '../wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.gender.number.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')
wsc_datapoints_rep = pd.DataFrame(np.repeat(wsc_datapoints.values,3,axis=0))
wsc_datapoints_rep.columns = wsc_datapoints.columns

#print(wsc_datapoints_repeated)


results_filename = sys.argv[1]
results = pd.read_csv(results_filename, sep=',', error_bad_lines=False, header=0)
new_texts = []

for survey_text in results.loc[:,'Input.text']:
    new_text = survey_text.replace("<font color='red'> <b>", '').replace("</b> </font>", '')
    new_text = re.sub(' +', ' ',new_text)
    new_texts.append(new_text)
results.insert(10, 'new_text', new_texts, allow_duplicates=True)

results = results.sort_values(by = 'new_text')

new_orig_texts = []
for orig_text in wsc_datapoints_rep.loc[:,'text_original']:
    new_text = re.sub(' +', ' ',orig_text)
    new_orig_texts.append(new_text)

wsc_datapoints_rep.insert(10, 'new_orig_text', new_orig_texts)

wsc_datapoints_rep = wsc_datapoints_rep.sort_values(by = 'new_orig_text')

correct = 0
total = 0

#for survey, gold_text in zip(results.sort_values(by = 'new_text').loc[:,'new_text'],  wsc_datapoints_repeated.sort_values(by ='new_orig_text').loc[:, 'new_orig_text']):
#    print(survey.split(), "### survey")
#    print(gold_text.split(), "### orig")
#    if survey.split() != gold_text.split():
#        print("ALARM")
#        break

votes = []
total = 0
for answer, gold in zip(results.sort_values(by = 'new_text').loc[:,'example.label'],  wsc_datapoints_rep.sort_values(by ='new_orig_text').loc[:, 'correct_answer']):
    votes.append(answer)
    if len(votes) == 3:
        majority = most_frequent(votes).replace('.', '').replace(' ', '')
        votes = []
        total = total + 1
        if gold.replace('.', '').replace(' ', '') == majority.replace('.', '').replace(' ', ''):
            correct = correct + 1


print(correct, "correct")
print(total, "total")

print(correct / total, "accuracy")