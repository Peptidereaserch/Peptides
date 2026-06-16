import pandas as pd

csv_files = ["FirstSet.csv", "SecondSet.csv", "ThirdSet.csv"]

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    
    if 'Spectrum Info' in df.columns:
        df = df.drop(columns=['Spectrum Info'])
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df.insert(0, 'ID', range(1, len(combined_df) + 1))

# Save to CSV
combined_df.to_csv("combined_with_IDs.csv", index=False)