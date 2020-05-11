#  The Sensitivity of Language Models and Humans to Winograd Schema Perturbations 

## Data
All perturbations are stored as a TSV in `data/final.tsv`. Each perturbation is referred to by a field beginning with `text`; `pron_index` refer to the index of the ambiguous pronoun. `answer` fields refer to the appropriate answer. New indices and answers are provided for perturbations that affect either of these; they are suffixed with the name of the perturbation.

## Models
To evaluate a pretrained model, simply run the relevant script. Different scripts for different language models take variations in tokenisation into account.

