import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("correlations.csv", index_col=0)
feature_order = [
    "AGEP", "SCHL", "RELP", "DIS", "SEX", "MIL", "MAR", "RAC1P",
    "DREM", "ESP", "ANC", "MIG", "DEAR", "CIT", "NATIVITY", "DEYE"
]

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


plt.title("Correlation Matrix", fontsize=14, pad=60)
plt.tight_layout()
plt.show()