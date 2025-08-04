import os
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from openmm import unit, LangevinIntegrator, app
from openmm.app import PDBFile, Simulation
from pdbfixer import PDBFixer
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import torch
import numpy as np
from rdkit import Chem
import warnings
from openmm import unit, Platform, State
from joblib import wrap_non_picklable_objects
from joblib import delayed
import re
from openmm.app import Modeller
import sys
from pdb_fixer import clean_structure,fix_pdb
from openmm.app.internal.pdbstructure import PdbStructure
import io
import subprocess
import csv
import copy
from Bio import PDB
import scipy.spatial as spatial
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

"Some scripts is based on https://github.com/CAODH/SurfDock/blob/master/force_optimize/minimize_utils.py"

def delete_mg_bond(protein_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(protein_path.replace('.pdb','_uff.pdb'))
    return protein_path.replace('.pdb','_uff.pdb')


def correct_uff(mol, protein_file, output_file):
    rd_mol = copy.deepcopy(mol)
    pkt_mol = Chem.MolFromPDBFile(protein_file)
    try:
        opt_mol = uff_geomopt(rd_mol,pkt_mol)
    except:
        uff_protein_file = delete_mg_bond(protein_file)
        pkt_mol = Chem.MolFromPDBFile(uff_protein_file)
        opt_mol = uff_geomopt(rd_mol,pkt_mol)
    return opt_mol  
    

def get_rd_atom_res_id(rd_atom):
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )


def get_pocket(lig_mol, rec_mol, max_dist=8):
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_mol.GetConformer().GetPositions()
    dist = spatial.distance.cdist(lig_coords, rec_coords)
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])
    pocket_res_ids = set()
    for i in pocket_atom_idxs:
        atom = rec_mol.GetAtomWithIdx(int(i))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)
    pkt_mol = rec_mol
    pkt_mol = Chem.RWMol(pkt_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())
    Chem.SanitizeMol(pkt_mol)
    return pkt_mol


def uff_geomopt(rd_mol, pkt_mol, lig_constraint=None, n_iters=20, n_tries=2, lig_h=False, pkt_h=False):
    pkt_mol = get_pocket(rd_mol, pkt_mol, max_dist=6)
    if lig_h:
        rd_mol = Chem.AddHs(rd_mol, addCoords=True)
    if pkt_h:
        pkt_mol = Chem.AddHs(pkt_mol, addCoords=True)
    rd_mol_rw = Chem.RWMol(rd_mol)
    uff_mol = Chem.CombineMols(pkt_mol, rd_mol_rw)
    Chem.SanitizeMol(uff_mol)
    uff = AllChem.UFFGetMoleculeForceField(uff_mol, confId=0, ignoreInterfragInteractions=False)
    uff.Initialize()
    for i in range(pkt_mol.GetNumAtoms()):
        uff.AddFixedPoint(i)
    if lig_constraint is not None:
        for i in lig_constraint:
            atom_idx_in_uff = pkt_mol.GetNumAtoms() + i 
            uff.AddFixedPoint(atom_idx_in_uff)
    converged = False 
    while n_tries > 0 and not converged:
        converged = not uff.Minimize(maxIts=n_iters)
        n_tries -= 1 
    conf = rd_mol_rw.GetConformer()
    for atom_idx in range(rd_mol.GetNumAtoms()):
        pos = uff_mol.GetConformer().GetAtomPosition(pkt_mol.GetNumAtoms() + atom_idx)
        conf.SetAtomPosition(atom_idx, pos)
    return rd_mol_rw


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))
    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)
        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')
        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        print("RDKit was unable to read the molecule:", e)
        return None
    return mol


def trySystem(system_generator,modeller,ligand_mol,pdb_id):
    max_attempts = 20
    attempts = 0
    success = False
    while attempts < max_attempts and not success:
        try:
            system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
            success = True  # Mark
        except Exception as e:
            # extract the error residue index from the error message
            # print(f'Try DELETE THIS ERROE {str(e)}!')
            match = re.search(r"residue (\d+)", str(e))
            if match:
                extracted_index = int(match.group(1)) - 1
                # located and record the residue to delete
                current_index = 0
                residue_to_delete = None
                for residue in modeller.topology.residues():
                    if current_index == extracted_index:
                        residue_to_delete = residue
                        break
                    current_index += 1

                modeller.delete([residue_to_delete])
        finally:
            attempts += 1
    if not success:
        print("Try maximum times but cannot create system")
        return None
    else:
        output_dir = os.path.join(os.getcwd(), 'modeller_systems')
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir,pdb_id+'_create_system.pdb'), "w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
    return modeller


def UpdatePose(ligand_mol,system_generator,modeller,protein_atoms,out_dir,pdb_id,receptor_path,device_num=0):
    try:
        # init save path
        out_base_dir = out_dir
        out_file = os.path.join(out_base_dir, pdb_id + '_minimized.sdf')
        if os.path.exists(out_file):
            return 0

        try: 
            lig_mol = Molecule.from_rdkit(ligand_mol,allow_undefined_stereo=True)
        except:
            return 2
        lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
        lig_top = lig_mol.to_topology()
        modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())
        # create simulation system
        try:
            system = system_generator.create_system(modeller.topology, molecules=lig_mol)
        except Exception as e:
            return 2
        # keep protein atom static in smiulation 
        num_atoms = system.getNumParticles()
        for atom in protein_atoms:
            if atom.index >= num_atoms:
                continue
            else:
                system.setParticleMass(atom.index, 0.000 * unit.dalton)
        # start simulation
        platform = GetPlatform()
        try:
            simulation, delta_energy = EnergyMinimized(modeller,system, platform,verbose=False,device_num=device_num)
        except:
            return 2

        
        ligand_atoms = list(filter(lambda atom:  atom.residue.name == 'UNK',list(modeller.topology.atoms())))
        ligand_index = [atom.index for atom in ligand_atoms]
        new_coords = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)[ligand_index]
        lig_mol = lig_mol.to_rdkit()
        conf = lig_mol.GetConformer()
        for i in range(lig_mol.GetNumAtoms()):
            x,y,z = new_coords.astype(np.double)[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        if delta_energy == 0.00 * unit.kilojoule / unit.mole:
            lig_mol = correct_uff(lig_mol, receptor_path, out_file)
        try:
            writer = Chem.SDWriter(out_file)
            writer.write(lig_mol)
            writer.close()
        except:
            out_base_dir = os.path.join(out_dir,pdb_id + '_tmp')
            os.makedirs(out_base_dir,exist_ok=True)
            out_file = os.path.join(out_base_dir,pdb_id + '_minimized.sdf')

            if os.path.exists(out_file):
                return 0
            writer = Chem.SDWriter(out_file)
            writer.write(lig_mol)
            writer.close()
        return 0
    except Exception as e:
        error_info = traceback.format_exc()
        print(f"Error details: {error_info}")
        print(f"Warning: {e}")
        with open('error_sdf.txt','a') as f:
            f.write(pdb_id +': error by :' + error_info + '\n')
        return 1



def GetFFGenerator(protein_forcefield = 'amber/ff14SB.xml',water_forcefield = 'amber/tip3p_standard.xml',small_molecule_forcefield = 'openff-2.0.0',ignoreExternalBonds=False):
    """
    Get forcefield generator by different forcefield files
    """
    forcefield_kwargs = {'constraints': None, 'rigidWater': True, 'removeCMMotion': False, 'ignoreExternalBonds': ignoreExternalBonds, 'hydrogenMass': 4*unit.amu }
    system_generator = SystemGenerator(
                forcefields=[protein_forcefield, water_forcefield ],
                small_molecule_forcefield=small_molecule_forcefield,
                forcefield_kwargs=forcefield_kwargs)
    return system_generator


def GetfixedPDB(receptor_path):
    temp_fixd_pdbs = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils/fixed_pdbs'
    os.makedirs(temp_fixd_pdbs,exist_ok=True)
    if not os.path.exists(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb'))):
        alterations_info  = {}
        fixed_pdb = fix_pdb(receptor_path, alterations_info)
        fixed_pdb_file = io.StringIO(fixed_pdb)
        pdb_structure = PdbStructure(fixed_pdb_file)
        clean_structure(pdb_structure, alterations_info)
        fixer = PDBFile(pdb_structure)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb')), 'w'))
    else:
        fixer = PDBFixer(os.path.join(temp_fixd_pdbs,os.path.basename(receptor_path).replace('.pdb','_fixer_processed_cleanup.pdb')))
    return fixer


@delayed
@wrap_non_picklable_objects
def GetPlatformPara():
    """Determine the best simulation platform available."""
    platform_name = os.getenv('PLATFORM')
    # properties = {'CudaDeviceIndex': '0'}
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
    else:
        platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())), key=lambda x: x.getSpeed())
    if platform.getName() in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
    return platform


def GetPlatform():
    """Determine the best simulation platform available."""
    platform_name = os.getenv('PLATFORM')
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)
    else:
        platform = max((Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())), key=lambda x: x.getSpeed())
    if platform.getName() in ['CUDA', 'OpenCL']:
        platform.setPropertyDefaultValue('Precision', 'mixed')
    return platform


def EnergyMinimized(modeller, system, platform, verbose=False,device_num=0):
    integrator = LangevinIntegrator(
    300 * unit.kelvin,
    1 / unit.picosecond,
    0.002 * unit.picoseconds)

    simulation = Simulation(modeller.topology, system = system, integrator = integrator, platform=platform) 
    simulation.context.setPositions(modeller.positions)
    while energy_penultimate > 30 * unit.kilojoule / unit.mole:
        simulation.minimizeEnergy(tolerance=0.1*unit.kilojoule/unit.mole/unit.nanometer,maxIterations=10)
        state_penultimate = simulation.context.getState(getEnergy=True)
        energy_penultimate = state_penultimate.getPotentialEnergy()
        simulation.minimizeEnergy(tolerance=0.1*unit.kilojoule/unit.mole/unit.nanometer,maxIterations=1)
        state_final = simulation.context.getState(getEnergy=True)
        energy_final = state_final.getPotentialEnergy()
        delta_energy = energy_penultimate - energy_final
    return simulation, delta_energy
