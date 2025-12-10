import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset_name = "ACSEmployment_size_var"

df = pd.read_csv(f"{dataset_name}_correlations.csv", index_col=0)
if dataset_name == "ACSEmployment":
    feature_order = ['AGEP','SCHL', 'DIS', 'RELP', 'SEX', 'MIL', 'MAR', 'RAC1P', 'ESP', 'DREM', 'ANC',
 'CIT', 'MIG', 'NATIVITY', 'DEYE', 'DEAR']
elif dataset_name == "ACSIncome":
    feature_order = ['SCHL', 'SEX', 'OCCP', 'AGEP', 'WKHP', 'RELP', 'RAC1P', 'MAR', 'COW', 'POBP']
elif dataset_name == "ACSPublicCoverage":
    feature_order = [
        'ESR', 'PINCP', 'DIS', 'SEX', 'SCHL', 'MAR', 'AGEP', 'MIL', 'FER', 'RAC1P',
        'ESP', 'MIG', 'DREM', 'ANC', 'DEAR', 'DEYE', 'CIT', 'NATIVITY']
elif dataset_name == "ACSEmployment_size_var":
    feature_order = ['AGEP', 'SCHL', 'DIS', 'RELP', 'MIL', 'SEX', 'MAR', 'DREM', 'ESP', 'RAC1P', 'ANC',
 'MIG', 'DEAR', 'CIT', 'DEYE', 'NATIVITY']


df = df.loc[feature_order]

rename_dict = {
    "feature_name": "Feature",
    "r_conflict_disagreement": "r(conflict,sign)",
    "r_var_disagreement": "r(var,sign)",
    "r_range_disagreement": "r(range,sign)",
    "r_conflict_variability": "r(conflict,var)",
    "r_conflict_range": "r(conflict,range)",
    "r_var_range": "r(var,range)",
}
df = df.rename(columns=rename_dict)

plt.rcParams.update({
    "font.size": 12,        # default text size
    "axes.labelsize": 14,   # x/y labels
    "axes.titlesize": 15,   # plot titles
    "xtick.labelsize": 12,  # x-tick labels
    "ytick.labelsize": 12,  # y-tick labels
    "legend.fontsize": 12
})
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    df.abs(),
    vmin=0, vmax=1,
    cmap="YlGn",
    annot=df,
    fmt=".4f",
)
cbar = ax.collections[0].colorbar
cbar.set_label("absolute correlation strength", fontsize=12)

ax.xaxis.set_ticks_position('top')
plt.ylabel("feature")
ax.xaxis.set_label_position('top')


plt.title(f"Correlation Matrix {dataset_name}", fontsize=14, pad=60)
plt.tight_layout()
plt.savefig(f"{dataset_name}_plots/{dataset_name}Correlations.png", dpi=300, bbox_inches="tight")
plt.close()