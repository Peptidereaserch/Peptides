import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# I trust you have the ability to chaange paths
INPUT_CSV = r"C:\Users\Waheg\Peptides\Peptide\combined_with_IDs.csv"
OUTPUT_CSV = r"C:\Users\Waheg\Peptides\Peptide\NewDatasetWithValues"

def NOMNOM(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x in ["—", "-", "nan", "None"]:
        return ""
    return x

def AddFeature(sequence, modifications):
    sequence = NOMNOM(sequence).upper().replace(" ", "")
    modifications = NOMNOM(modifications)
    features = {}

    if sequence == "":
        return features
    try:
        peptide = ProteinAnalysis(sequence)
        features["length"] = len(sequence)
        features["molecular_weight"] = peptide.molecular_weight()
        features["charge_pH_7"] = peptide.charge_at_pH(7.0)
        features["hydrophobicity_GRAVY"] = peptide.gravy()
        features["isoelectric_point"] = peptide.isoelectric_point()
        features["oxidation_flag"] = 1 if "oxidation" in modifications.lower() else 0
        aa_counts = peptide.count_amino_acids()
        aa_percent = peptide.get_amino_acids_percent()
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            features[f"{aa}_count"] = aa_counts.get(aa, 0)
            features[f"{aa}_percent"] = aa_percent.get(aa, 0)

# this except is broken 
    except Exception as e:
        features["feature_error"] = str(e)
    return features

# I am an Inter to do anything like this main 
def main():
    df = pd.read_csv(INPUT_CSV)
    all_features = []

    for _, row in df.iterrows():
        sequence = row.get("Peptide Sequence", "")
        modifications = row.get("Peptide Modifications", "")
        features = AddFeature(sequence, modifications)
        all_features.append(features)

    FeatureDataset= pd.DataFrame(all_features)
    Final = pd.concat([df, FeatureDataset], axis=1)
    Final.to_csv(OUTPUT_CSV, index=False)
    print("New file created successfully!")
    print("Saved to:", OUTPUT_CSV)
    print("Rows:", len(Final))
    print("Columns:", len(Final.columns))
if __name__ == "__main__":
    main()