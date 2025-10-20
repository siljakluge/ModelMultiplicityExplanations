import pandas as pd

df = pd.read_csv("correlations.csv").round(4)
rename_dict = {
    "feature_name": "Feature",
    "r_conflict_disagreement": "r(conflict,dis)",
    "r_var_disagreement": "r(var,dis)",
    "r_range_disagreement": "r(range,dis)",
    "r_conflict_variability": "r(conflict,var)",
    "r_conflict_range": "r(conflict,range)",
}
df = df.rename(columns=rename_dict)
latex_table = df.to_latex(
    index=False,
    escape=False,
    float_format="%.4f",
    caption="Correlations between conflict rate, disagreement, variability, and range.",
    label="tab:corr_summary"
)
print(latex_table)