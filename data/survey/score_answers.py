import pandas as pd
import sys

path_to_wsc = '../wsc_data/enhanced.tense.random.role.syn.voice.scramble.freqnoun.tsv'
wsc_datapoints = pd.read_csv(path_to_wsc, sep='\t')
print(wsc_datapoints)


results_filename = sys.argv[1]
results = pd.read_csv(results_filename, sep='delimiter', error_bad_lines=False, header=0)


participant_results = results.iloc[4, :]


correct = 0
total = 0
for answer, gold in zip(participant_results[0].split(',')[1:-4],  wsc_datapoints.loc[:, 'correct_answer']):
    total = total + 1
    print(answer, gold.replace('.', '').replace(' ', ''))
    if answer == str(1) and gold.replace('.', '').replace(' ', '') == 'A':
        correct = correct + 1
    if answer == str(2) and gold.replace('.', '').replace(' ', '') == 'B':
        correct = correct + 1


print(correct, "correct")
print(total, "total")

print(correct / total, "accuracy")

