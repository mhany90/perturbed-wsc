import sys

filename = sys.argv[1]
file = open(filename, 'r').read()
dps = file.split("\n\n")
no_dps = len(dps)
split_counts = 4
split_size = no_dps / 4

split_counter = 0
for index, dp in enumerate(dps):
    if index % 71 == 0 or index == 0: 
        split_counter = split_counter + 1
        fsplitname = str(filename) + "." + str(split_counter)
        if not index == 0:
            f.close()
        f = open(fsplitname, 'w')
    f.write(dp)
    f.write("\n")
 
