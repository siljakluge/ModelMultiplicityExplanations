import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_name = "ACSEmployment_sizes"

df = pd.read_csv(f"{dataset_name}_correlations.csv", index_col=0)
if dataset_name == "ACSEmployment":
    feature_order = [
        "AGEP", "SCHL", "RELP", "DIS", "SEX", "MIL", "MAR", "RAC1P",
        "DREM", "ESP", "ANC", "MIG", "DEAR", "CIT", "NATIVITY", "DEYE"
    ]
elif dataset_name == "ACSIncome":
    feature_order = ['SCHL', 'SEX', 'WKHP', 'OCCP', 'AGEP', 'RELP', 'RAC1P', 'MAR', 'COW', 'POBP']
elif dataset_name == "ACSPublicCoverage":
    feature_order = [
        'ESR', 'PINCP', 'DIS', 'SEX', 'SCHL', 'MAR', 'AGEP', 'MIL', 'FER', 'RAC1P',
        'ESP', 'MIG', 'DREM', 'ANC', 'DEAR', 'DEYE', 'CIT', 'NATIVITY', 'ST'
    ]
elif dataset_name == "ACSEmployment_sizes":
    feature_order = ['AGEP', 'SCHL', 'DIS', 'RELP', 'MIL', 'SEX', 'MAR', 'DREM', 'ESP', 'RAC1P', 'ANC',
 'MIG', 'DEAR', 'CIT', 'DEYE', 'NATIVITY']

df = df.loc[feature_order]

rename_dict = {
    "feature_name": "Feature",
    "r_conflict_disagreement": "r(conflict,dis)",
    "r_var_disagreement": "r(var,dis)",
    "r_range_disagreement": "r(range,dis)",
    "r_conflict_variability": "r(conflict,var)",
    "r_conflict_range": "r(conflict,range)",
    "r_var_range": "r(var,range)",
}
df = df.rename(columns=rename_dict)

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    df.abs(),
    vmin=0, vmax=1,
    cmap="YlGn",
    annot=df,
    fmt=".4f",
)

# Achsenbeschriftungen nach oben verschieben
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')


plt.title(f"Correlation Matrix {dataset_name}", fontsize=14, pad=60)
plt.tight_layout()
plt.show()