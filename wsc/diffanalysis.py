import numpy as np
import torch
import pickle
from helpers import safe_increment, safe_append
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt

NUM_LAYERS, NUM_HEADS = 24, 16

with open('attention_div.dump', 'rb') as f:
    packages = pickle.load(f)

k1, k2 = 'text_original', 'text_syn'
exp_name = k2.split('_')[1]
d1, d2 = packages[k1], packages[k2]
assert len(d1) == len(d2)

valid_keys = [n for (n, i) in enumerate(d2) if i]
d1, d2 = np.array(d1), np.array(d2)
orig, pert = d1[valid_keys], d2[valid_keys]

diffs = torch.zeros(NUM_LAYERS, NUM_HEADS)
answer_dict = [[{} for i in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]
labels = [[0 for i in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]
for n, (o, p) in enumerate(zip(orig, pert)):
    if not o or not p:
        continue

    o_attn, o_tok, (o_o, o_x, o_d) = o
    p_attn, p_tok, (p_p, p_x, p_d) = p
    if o_attn.shape[3] != p_attn.shape[3]:
        continue

    p_attn = p_attn.mean(dim=-2)
    o_attn = o_attn.mean(dim=-2)

    var = (p_attn - o_attn)/ o_attn

    a, t = var.abs().topk(dim=-1, k=var.shape[-1])
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            for _a, _t in zip(a[layer][head], t[layer][head]):
                token = o_tok[_t]
                if token in ['en', 'long']:
                    continue
                safe_append(answer_dict[layer][head], token, _a.item())


try:
    with open('mapdict_{}.pickle'.format(exp_name), 'rb') as f:
        mapdict = pickle.load(f)
        mapdict['around'] = 'prep'
except:
    mapdict = {}

for layer in range(NUM_LAYERS):
    for head in range(NUM_HEADS):
        for k in answer_dict[layer][head].keys():
            answer_dict[layer][head][k] = \
                np.mean(answer_dict[layer][head][k]) if len(answer_dict[layer][head][k]) > 0 else float('-inf')

        m = max(answer_dict[layer][head].items(), key=lambda x: x[1])
        ### BUILDER
        answer_dict[layer][head] = m[1]
        # labels[layer][head] = m[0]
        if m[0] not in mapdict.keys():
            s = input(m[0])
            labels[layer][head] = s
            mapdict[m[0]] = s
        else:
            labels[layer][head] = mapdict[m[0]]

with open('mapdict_{}.pickle'.format(exp_name), 'wb') as f:
    pickle.dump(mapdict, f)

vals = pd.DataFrame(answer_dict)
labels = pd.DataFrame(labels)
ax = sns.heatmap(vals, annot=labels, fmt="", cmap='Greys')
ax.set(xlabel='Heads', ylabel='Layers')
ax.invert_yaxis()
plt.show()
