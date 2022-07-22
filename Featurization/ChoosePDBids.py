import numpy as np

# Read 'entry_ids' into array.
# Total number of recorded proteins is 191869.
NUM = 191869
with open('entry_ids') as f:
    ids = f.readlines()
    ids = np.array(ids[0][1:-1].split(","))
    # ids: a list containing str ids for each protein
    # ex: ['"140D"', '"1QFN"', ...]
    num = len(ids)
    assert num == NUM, "Check your bug! There should be %s in total." % (NUM)

# Randomly choose some ids from the array.
np.random.seed(0)
n_tobe_chosen = 2000
ids_tobe_chosen = np.random.choice(num, n_tobe_chosen, replace=False)
chosen = ids[ids_tobe_chosen]

# Output the chosen ids into txt.
f = open("./IntermediateData/chosen_ids.txt", "w")
for i in range(n_tobe_chosen):
    f.write(chosen[i][1:-1])
    if i < n_tobe_chosen - 1:
        f.write(", ")
f.close()

