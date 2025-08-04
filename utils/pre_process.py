
import os
from multiprocessing import Pool
import MDAnalysis as mda
import argparse
from functools import partial 


argparser = argparse.ArgumentParser()
argparser.add_argument('--complex_path', type=str, default='../dataset/metallo_complex')                                       
argparser.add_argument('--pdbid_list', nargs='+', default=[])
args = argparser.parse_args()

def generate_pocket(pdb_file, ligand_file, output_pdb, include_ligand=True, distance_cutoff=10.0):
    u = mda.Universe(pdb_file)
    l = mda.Universe(ligand_file)
    ligand = l.select_atoms("all")
    
    ligand_name = l.atoms[0].resname
    ligand_chain = l.atoms[0].segid
    ligand_resid = l.atoms[0].resid
    for res in u.residues:
        if res.resname == ligand_name:
            matching_ligand = res
            matching_ligand_atoms = matching_ligand.atoms
            break
    pocket_atoms = u.select_atoms(f"around 10. group ligand", ligand=matching_ligand_atoms)
    hatom = pocket_atoms.select_atoms(f"element H")
    pocket_atoms = pocket_atoms.atoms - hatom
    if not include_ligand:
        pocket_atoms = pocket_atoms.select_atoms(f'not (resname {ligand_name})')
    else:
        pocket_atoms += matching_ligand.atoms
    pocket_residue = pocket_atoms.residues
    pocket = pocket_residue.atoms
    
    if len(pocket) == 0:
        raise ValueError(f"No atoms found within {distance_cutoff} Å of the {os.path.basename(pdb_file)}.")
    pocket.write(output_pdb)



def preprocess_pipeline(id, path):
    pdbid = id
    complex_pdb_file = os.path.join(path, pdbid, pdbid+'_prepare.pdb') 
    ligand_file = os.path.join(path, pdbid, pdbid+'_ligand.pdb')
    protein_output_pdb = os.path.join(path, pdbid, pdbid+'_pro_pocket.pdb')
    complex_output_pdb = os.path.join(path, pdbid, pdbid+'_com_pocket.pdb')
    generate_pocket(complex_pdb_file, ligand_file, complex_output_pdb, include_ligand=True, distance_cutoff=10.0)
    generate_pocket(complex_pdb_file, ligand_file, protein_output_pdb, include_ligand=False, distance_cutoff=10.0)




if __name__ == '__main__':
    single_process = partial(preprocess_pipeline, path=args.complex_path)
    pool = Pool()
    pool.map(single_process, args.pdbid_list)
    pool.close()
    pool.join()



