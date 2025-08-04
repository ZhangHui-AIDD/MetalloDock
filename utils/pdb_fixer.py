'''Copyright 2021 DeepMind Technologies Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import io
from typing import Dict, Set, List, Any

from pdbfixer import PDBFixer
from openmm import app
from openmm.app import element
from openmm.app.internal import pdbstructure


class StructureProcessor:
    """
    A comprehensive processor for PDB structure manipulation and fixing.
    Handles structure parsing, cleaning, and various molecular corrections.
    """
    
    def __init__(self):
        self.modification_log = {}
    
    def pdb_to_structure(self, pdb_content: str) -> pdbstructure.PdbStructure:
        """
        Convert PDB string content to OpenMM PdbStructure object.
        
        Args:
            pdb_content: Raw PDB file content as string
            
        Returns:
            PdbStructure object for further processing
        """
        string_buffer = io.StringIO(pdb_content)
        return pdbstructure.PdbStructure(string_buffer)
    
    def fix_pdb(self, input_pdb_file, structural_changes_log: Dict[str, Any]) -> str:
        """
        Apply comprehensive PDB structure fixes using PDBFixer.
        
        Performs the following operations:
        - Replaces nonstandard residues with standard equivalents
        - Removes heterogens (non-protein residues) including water molecules
        - Adds missing residues and atoms within existing residues
        - Adds hydrogen atoms assuming physiological pH (7.0)
        - Maintains original chain and residue identifiers
        
        Args:
            input_pdb_file: Input PDB file handle or path
            structural_changes_log: Dictionary to record all modifications made
            
        Returns:
            Cleaned PDB structure as string format
        """
        structure_fixer = PDBFixer(input_pdb_file)
        
        # Handle nonstandard residues
        structure_fixer.findNonstandardResidues()
        structural_changes_log['nonstandard_residues'] = structure_fixer.nonstandardResidues
        structure_fixer.replaceNonstandardResidues()
        
        # Remove heterogeneous molecules
        self._eliminate_heterogens(structure_fixer, structural_changes_log, retain_water=False)
        
        # Add missing structural components
        structure_fixer.findMissingResidues()
        structural_changes_log['missing_residues'] = structure_fixer.missingResidues
        structure_fixer.findMissingAtoms()
        structural_changes_log['missing_heavy_atoms'] = structure_fixer.missingAtoms
        structural_changes_log['missing_terminals'] = structure_fixer.missingTerminals
        structure_fixer.addMissingAtoms(seed=0)
        structure_fixer.addMissingHydrogens()
        
        # Generate output PDB string
        output_buffer = io.StringIO()
        app.PDBFile.writeFile(
            structure_fixer.topology, 
            structure_fixer.positions, 
            output_buffer,
            keepIds=True
        )
        return output_buffer.getvalue()
    
    def clean_structure(self, protein_structure, structural_changes_log: Dict[str, Any]) -> None:
        """
        Apply additional structure refinements to handle edge cases.
        
        Args:
            protein_structure: OpenMM structure object to modify
            structural_changes_log: Dictionary to record modifications
        """
        self._correct_selenomethionine(protein_structure, structural_changes_log)
        self._remove_single_residue_chains(protein_structure, structural_changes_log)
    
    def _eliminate_heterogens(self, structure_fixer: PDBFixer, 
                            structural_changes_log: Dict[str, Any], 
                            retain_water: bool) -> None:
        """
        Remove heterogeneous molecules from the structure.
        
        Args:
            structure_fixer: PDBFixer instance to modify
            structural_changes_log: Dictionary to record removed heterogens
            retain_water: Whether to preserve water molecules (HOH)
        """
        initial_residue_types = set()
        for protein_chain in structure_fixer.topology.chains():
            for residue_unit in protein_chain.residues():
                initial_residue_types.add(residue_unit.name)
        
        structure_fixer.removeHeterogens(keepWater=retain_water)
        
        final_residue_types = set()
        for protein_chain in structure_fixer.topology.chains():
            for residue_unit in protein_chain.residues():
                final_residue_types.add(residue_unit.name)
        
        structural_changes_log['removed_heterogens'] = (
            initial_residue_types.difference(final_residue_types)
        )
    
    def _correct_selenomethionine(self, protein_structure, 
                                structural_changes_log: Dict[str, Any]) -> None:
        """
        Replace selenium atoms with sulfur in methionine residues.
        
        Corrects selenomethionine (SeMet) residues that were not properly
        identified as modified residues during initial processing.
        
        Args:
            protein_structure: OpenMM structure to modify
            structural_changes_log: Dictionary to record selenium corrections
        """
        corrected_met_residues = []
        
        for residue_unit in protein_structure.iter_residues():
            residue_name = residue_unit.get_name_with_spaces().strip()
            if residue_name == 'MET':
                sulfur_atom = residue_unit.get_atom('SD')
                if sulfur_atom.element_symbol == 'Se':
                    sulfur_atom.element_symbol = 'S'
                    sulfur_atom.element = element.get_by_symbol('S')
                    corrected_met_residues.append(sulfur_atom.residue_number)
        
        structural_changes_log['Se_in_MET'] = corrected_met_residues
    
    def _remove_single_residue_chains(self, protein_structure, 
                                    structural_changes_log: Dict[str, Any]) -> None:
        """
        Remove chains containing only a single amino acid residue.
        
        Single residue chains are problematic as they are both N and C terminus
        simultaneously, which lacks proper force field templates.
        
        Args:
            protein_structure: OpenMM structure to modify
            structural_changes_log: Dictionary to record removed chains
        """
        eliminated_chains = {}
        
        for structural_model in protein_structure.iter_models():
            valid_chain_list = [
                chain for chain in structural_model.iter_chains() 
                if len(chain) > 1
            ]
            invalid_chain_identifiers = [
                chain.chain_id for chain in structural_model.iter_chains() 
                if len(chain) <= 1
            ]
            
            structural_model.chains = valid_chain_list
            
            for chain_identifier in invalid_chain_identifiers:
                structural_model.chains_by_id.pop(chain_identifier)
            
            eliminated_chains[structural_model.number] = invalid_chain_identifiers
        
        structural_changes_log['removed_chains'] = eliminated_chains


# Convenience functions for backward compatibility
def pdb_to_structure(pdb_str: str) -> pdbstructure.PdbStructure:
    """Convert PDB string to structure - backward compatibility wrapper."""
    processor = StructureProcessor()
    return processor.pdb_to_structure(pdb_str)


def fix_pdb(pdbfile, alterations_info: Dict[str, Any]) -> str:
    """Fix PDB structure - backward compatibility wrapper."""
    processor = StructureProcessor()
    return processor.fix_pdb(pdbfile, alterations_info)


def clean_structure(pdb_structure, alterations_info: Dict[str, Any]) -> None:
    """Clean structure - backward compatibility wrapper."""
    processor = StructureProcessor()
    processor.clean_structure(pdb_structure, alterations_info)


