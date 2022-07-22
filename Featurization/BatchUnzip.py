import os
import gzip
import shutil


'''
Solutions from stackoverflow:
https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
'''
# Unzip the .gz files.
path = './IntermediateData/pdb_data_2000'
for fname in os.listdir(path):
    if fname[-3:] == '.gz':
        with gzip.open(os.path.join(path, fname), 'rb') as f_in:
            with open(os.path.join(path, fname[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Then remove .gz files and some other irrelevant files.
for fname in os.listdir(path):
    if fname[-3:] == '.gz' or fname == '.DS_Store' or fname == '.ipynb_checkpoints':
        os.remove(os.path.join(path, fname))

# Check the number.
cnt = 0
for fname in os.listdir(path):
    if fname[-4:] == '.pdb':
        cnt += 1
print(f"The total extracted pdb files: {cnt}.")
print(f"Some pdb ids on the entry_ids list are missing.")
    

