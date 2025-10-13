import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import os
import pandas as pd

from folktables import ACSDataSource, ACSEmployment

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)
feature_names = ACSEmployment.features

# anpassen:
n_models = 10
n_datapoints = 500
n_shap_samples = 100
no_conflict_shap = True
save_data = False
load_previous_data = False

def create_mlp():
    classifier = MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True)
    return make_pipeline(StandardScaler(), classifier)

def rashomon_set(n_models=50, tolerance=0.01, base_seed=11):
    X_tr, X_test, y_tr, y_test, group_train, group_test = train_test_split(
        features, label, group, test_size=0.2, random_state=0)
    rng = np.random.RandomState(base_seed)
    random_seeds = rng.randint(0, 10_000, size=n_models)

    models, accuracies, train_accs = [], [], []
    for seed in random_seeds:
        pipe = create_mlp()
        pipe.set_params(mlpclassifier__random_state=seed)
        pipe.fit(X_tr, y_tr)

        acc_score = accuracy_score(y_test, pipe.predict(X_test))
        train_score = round(pipe.score(X_tr, y_tr), 4)
        models.append(pipe)
        accuracies.append(acc_score)
        train_accs.append(train_score)

    print("Training Accuracies: ", train_accs)
    accs = np.array(accuracies)
    best_acc = accs.max()
    threshold = best_acc - tolerance
    keep = accs >= threshold
    good_models = [m for m, k in zip(models, keep) if k]

    return good_models, (X_test, y_test), best_acc


def ambiguity(classifiers, points):
    if len(classifiers) < 2:
        return 0.0
    predictions = np.vstack([m.predict(points) for m in classifiers])
    h_0 = predictions[0]
    disagrees = np.any(predictions != h_0, axis=0)
    return float(disagrees.mean())

# Konfliktrate
def conflict_rate(classifiers, points):
    preds = np.vstack([m.predict(points) for m in classifiers])
    p = preds.mean(axis=0)
    rate = np.minimum(p, 1 - p)
    conflict_indices = np.where(rate > 0)[0]
    return rate, conflict_indices

if load_previous_data:
    shap_values_all = np.load("previous data/shap_values_all.npy")
    final_rate = np.load("previous data/conflict_rate.npy")
else:
    rashomon_models, (X_val, y_val), acc = rashomon_set(n_models=n_models, tolerance=0.015, base_seed=11)
    print("Models:", len(rashomon_models))
    print("Best Accuracy:", acc.round(4))
    print("Ambiguity:", round(ambiguity(rashomon_models, X_val), 4))

    idx = np.random.choice(len(X_val), size=n_datapoints, replace=False)
    datapoint_subset = X_val[idx]
    rate, indices = conflict_rate(rashomon_models, datapoint_subset)

    final_rate = rate[indices]
    if save_data:
        os.makedirs("previous data", exist_ok=True)
        np.save("previous data/conflict_rate.npy", final_rate)
    conflict_data = datapoint_subset[indices]
    print("Number of conflicting points:", len(conflict_data))

    # noc conflict points
    all_indices = np.arange(len(datapoint_subset))
    non_conflict_indices = np.setdiff1d(all_indices, indices)
    non_conflict_data = datapoint_subset[non_conflict_indices]

    # SHAP
    sample = shap.sample(X_val, n_shap_samples)

    shap_values_all = []
    for model in rashomon_models:
        explainer = shap.KernelExplainer(model.predict, sample)
        shap_values = explainer.shap_values(conflict_data)
        shap_values_all.append(shap_values)

    if save_data:
        np.save("previous data/shap_values_all.npy", shap_values_all)

    shap_values_all = np.array(shap_values_all)
    print(shap_values_all.shape)

# VZ-Wechsel
has_pos = (shap_values_all > 0).any(axis=0)
has_neg = (shap_values_all < 0).any(axis=0)
sign_change = has_pos & has_neg

# Range:
shap_min = shap_values_all.min(axis=0)
shap_max = shap_values_all.max(axis=0)
feature_ranges = shap_max - shap_min
max_range = feature_ranges.max(axis=1).max()
print("Average range per feature:", feature_ranges.mean(axis=0))
print("Max SHAP Range: ", round(max_range, 4))

# abs max SHAP Value pro Feature
shap_min_feature = shap_min.min(axis=0)
shap_max_feature = shap_max.max(axis=0)
abs_max_feature = np.maximum(np.abs(shap_min_feature), np.abs(shap_max_feature))
print("abs max feature: ", abs_max_feature)

# Variability
explanation_var = shap_values_all.var(axis=0)
max_var = explanation_var.max(axis=1).max()
print("Max Var: ", round(max_var, 4))

#Plots
os.makedirs("plots_VAR", exist_ok=True)
os.makedirs("plots_RANGE", exist_ok=True)

# Explanation Range Plot
for feat_idx in range(feature_ranges.shape[1]):
    correlation = np.corrcoef(final_rate, feature_ranges[:, feat_idx])[0, 1]
    colors = np.where(sign_change[:, feat_idx], "red", "green")

    plt.figure()
    plt.scatter(final_rate, feature_ranges[:, feat_idx], c=colors, alpha=0.5)
    plt.xlabel("Conflict Rate")
    plt.ylabel("SHAP Explanation Range")
    plt.xlim(0, 0.5)
    plt.ylim(0, max_range)
    #plt.ylim(0, abs_max_feature[feat_idx])
    plt.title(f"Feature {feature_names[feat_idx]}, (r = {correlation:.3f})")

    ax = plt.gca()
    ax.scatter([], [], c="red", label="Vorzeichenwechsel")
    ax.scatter([], [], c="green", label="konstantes Vorzeichen")
    ax.legend(loc="upper right", frameon=True)

    filename = f"plots_RANGE/{feat_idx}_{feature_names[feat_idx]}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# Explanation Variability Plot
for feat_idx in range(explanation_var.shape[1]):
    correlation = np.corrcoef(final_rate, explanation_var[:, feat_idx])[0, 1]
    colors = np.where(sign_change[:, feat_idx], "red", "green")

    plt.figure()
    plt.scatter(final_rate, explanation_var[:, feat_idx], c=colors, alpha=0.5)
    plt.xlabel("Conflict Rate")
    plt.ylabel("SHAP Explanation variability")
    plt.xlim(0, 0.5)
    plt.ylim(0, max_var)
    plt.title(f"Feature {feature_names[feat_idx]}, (r = {correlation:.3f})")

    ax = plt.gca()
    ax.scatter([], [], c="red", label="Vorzeichenwechsel")
    ax.scatter([], [], c="green", label="konstantes Vorzeichen")
    ax.legend(loc="upper right", frameon=True)

    filename = f"plots_VAR/{feat_idx}_{feature_names[feat_idx]}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# SHAP + Plot for no-conflict data:
if no_conflict_shap & (not load_previous_data):
    idx = np.random.choice(len(non_conflict_indices), size=100, replace=False)
    subset = non_conflict_data[idx]

    shap_values_all_no_conflict = []
    for model in rashomon_models:
        explainer = shap.KernelExplainer(model.predict, sample)
        shap_values = explainer.shap_values(subset)
        shap_values_all_no_conflict.append(shap_values)

    mean_abs_shap_per_feature = np.mean(np.abs(shap_values_all_no_conflict), axis=(0, 1))
    mean_abs_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap_per_feature
    }).sort_values("mean_abs_shap", ascending=False)

    plt.figure()
    plt.bar(mean_abs_df["feature"], mean_abs_df["mean_abs_shap"])
    plt.xticks(rotation=90)
    plt.ylabel("Mean absolute SHAP")
    plt.title("Mean absolute SHAP per feature for non-conflict points")
    os.makedirs("plots_MEAN_ABS", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots_MEAN_ABS/mean_abs_shap_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()

