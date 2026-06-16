import pandas as pd 
import requests
import time
import os
import numpy as np
import py3Dmol
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

path = "./psms.csv"
data = pd.read_csv(path)
print(data.head())

# testing some META Ai Stuff. 

peptide_sequence = data.iloc[6]['Peptide Sequence'] # Grabbing the Peptide Sequence

previous_peptide = None

output_dir = "Data/peptide_structures" # Directory to save PDB files
os.makedirs(output_dir, exist_ok=True) # Making the DIR
pdb_filename = os.path.join(output_dir, f"{peptide_sequence}.pdb") # Making the .PDB filename according to the peptide sequence name. 

# Call API

def fetch_peptide_structure(sequence, filename):  # Basic Function just calls url, deos a try catch to call the API. If it works then we just continue on. 
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    try: 
        response = requests.post(url, data=sequence, verify=False, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            return None
        
    except Exception as e:
        print(f"Request failed: {e}")
        return None
    
if previous_peptide:
    print(previous_peptide)
print(f"Fetching structure for peptide: {peptide_sequence}")
previous_peptide = peptide_sequence
pdb_data = fetch_peptide_structure(peptide_sequence, pdb_filename)

if pdb_data:  # If the data exists then we save the data to pdb_filename. 
    with open(pdb_filename, "w") as f:
        f.write(pdb_data)
    print(f"Success! Saved to {pdb_filename}")

# Next is the module to view this data. 
    # 4. VISUALIZE: Show 3D structure in VS Code
    view = py3Dmol.view(width=400, height=300)
    view.addModel(pdb_data, "pdb")
    view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}}) #This is the function to change the style. Just search Py3DMol to edit it. 
    view.zoomTo()
    # view.show() DOESN'T WORK IN A .py FILE

    # 5. ANALYZE: Calculate "Qualities" 
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_filename)

    # A. Radius of Gyration 
    # Calculate center of mass
    atoms = list(structure.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms])
    center_of_mass = coords.mean(axis=0)
    # Calculate Rg
    rg = np.sqrt(np.sum((coords - center_of_mass)**2) / len(atoms))
    
    # B. SASA (
    sr = ShrakeRupley()
    sr.compute(structure, level="S") # S = Structure level
    total_sasa = structure.sasa
    
    # C. Confidence (pLDDT)
    # ESMFold stores confidence in the B-factor column (last column of PDB)
    b_factors = [atom.bfactor for atom in atoms]
    avg_plddt = sum(b_factors) / len(b_factors)
    print("-" * 30)
    print(f"FEATURE REPORT FOR {peptide_sequence}")
    print("-" * 30)
    print(f"1. Confidence (pLDDT): {avg_plddt:.2f} (0-100, >70 is good)")
    print(f"2. Radius of Gyration: {rg:.2f} Å (Lower = More Compact)")
    print(f"3. Total Surface Area: {total_sasa:.2f} Å²")
    print("-" * 30)
else:
    print("Failed to generate structure.")
