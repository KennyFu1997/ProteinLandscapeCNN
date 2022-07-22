from Bio.PDB import PDBParser
from Bio.PDB.vectors import rotaxis
from Bio.PDB.Polypeptide import is_aa

import numpy as np
import os
import math


global PATH
PATH = './IntermediateData/pdb_data_2000'
global DICT_KEYS
DICK_KEYS = ('atom_name', 'coords_x', 'coords_y', 'coords_z', 'parent', 'resseq')
global OUTPATH1
OUTPATH1 = './IntermediateData/meta_data.npy'
global OUTPATH2
OUTPATH2 = './IntermediateData/meta_data_name.npy'
global DIC 
DIC = ['N', 'CA', 'C', 'O']

def process_atom(atom):
    '''
    input: Atom object
    output: a list of collected info
    '''
    atom_dict = dict.fromkeys(DICK_KEYS)
    parent_residue = atom.get_parent()
    
    atom_dict['atom_name'] = atom.name
    atom_dict['coords_x'] = atom.get_coord()[0]
    atom_dict['coords_y'] = atom.get_coord()[1]
    atom_dict['coords_z'] = atom.get_coord()[2]
    atom_dict['parent'] = parent_residue.get_resname()
    atom_dict['resseq'] = parent_residue.get_id()[1]
    
    atom_info = [atom_dict[key] for key in DICK_KEYS]
    
    return atom_info

def add_virtual_cb(residue):
    
    n = residue['N'].get_vector()
    c = residue['C'].get_vector()
    ca = residue['CA'].get_vector()
    n -= ca
    c -= ca
    rot = rotaxis(-math.pi*120.0/180.0, c)
    cb_at_origin = n.left_multiply(rot)
    cb = cb_at_origin+ca
    
    atom_dict = dict.fromkeys(DICK_KEYS)
    atom_dict['atom_name'] = 'CB'
    atom_dict['coords_x'] = cb[0]
    atom_dict['coords_y'] = cb[1]
    atom_dict['coords_z'] = cb[2]
    atom_dict['parent'] = res.get_resname()
    atom_dict['resseq'] = res.get_id()[1]
    
    atom_info = [atom_dict[key] for key in DICK_KEYS]
    
    return atom_info

prots_info = []
prots_name = []
# update: only process atoms on chain A and model 0
# update: some LYS doesn't have CB
for fname in os.listdir(PATH):
    pdb_id = fname[:-4]
    structure = PDBParser().get_structure(pdb_id, os.path.join(PATH, fname))
    atoms_info = []
    for res in structure.get_residues():
        mk1 = res.get_full_id()[2] == 'A'
        mk2 = res.get_full_id()[1] == 0
        mk3 = res.get_full_id()[3][0] == ' '
        mk4 = res.get_full_id()[3][2] == ' '
        
        if mk1 * mk2 * mk3 * mk4 * is_aa(res):
            # remove special residues
            test_validity = [key in res.child_dict.keys() for key in DIC]
            if sum(test_validity) != 4:
                continue
            CB_flag = False
            for atom in res.get_atoms():
                if atom.name == 'CB':
                    CB_flag = True
                atom_info = process_atom(atom)
                atoms_info.append(atom_info)
            if CB_flag == False:
                atom_info = add_virtual_cb(res)
                atoms_info.append(atom_info)
    atoms_info = np.array(atoms_info, dtype=object)
    if len(atoms_info) == 0:
        continue
    prots_info.append(atoms_info)
    prots_name.append(pdb_id)
prots_info = np.array(prots_info, dtype=object)
prots_name = np.array(prots_name, dtype=object)

np.save(OUTPATH1, prots_info)
np.save(OUTPATH2, prots_name)

