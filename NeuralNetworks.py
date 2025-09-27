import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

from folktables import ACSDataSource, ACSEmployment

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["AL"], download=True)
features, label, group = ACSEmployment.df_to_numpy(acs_data)
feature_names = ACSEmployment.features

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

    print(train_accs)
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

rashomon_models, (X_val, y_val), acc = rashomon_set(n_models=5, tolerance=0.015, base_seed=11)
print("Models:", len(rashomon_models))
print("Best Accuracy:", acc.round(4))
print("Ambiguity:", round(ambiguity(rashomon_models, X_val), 4))

datapoint_subset = X_val[:100]
rate, indices = conflict_rate(rashomon_models, datapoint_subset)

final_rate = rate[indices]
conflict_data = X_val[indices]
print("Number of conflicting points:", len(conflict_data))

# SHAP
sample = shap.sample(X_val, 50)

shap_values_all = []
for model in rashomon_models:
    explainer = shap.KernelExplainer(model.predict_proba, sample)
    shap_values = explainer.shap_values(conflict_data)

    # API workaround with ChatGPT:
    # handle both SHAP APIs
    if isinstance(shap_values, list) or (hasattr(shap_values, '__len__') and hasattr(shap_values, 'append') and not hasattr(shap_values, 'values')):
        # old API: list per class
        shap_pos = shap_values[1]  # (N, F)
    else:
        # new API: Explanation with .values of shape (N, F, C)
        vals = getattr(shap_values, "values", shap_values)  # some versions return the array directly
        shap_pos = vals[..., 1]  # (N, F)  <-- select positive class

    shap_values_all.append(shap_pos)

shap_values_all = np.array(shap_values_all)

# Range:
shap_min = shap_values_all.min(axis=0)
shap_max = shap_values_all.max(axis=0)
feature_ranges = shap_max - shap_min
max_range = feature_ranges.max(axis=1).max()
print("Max SHAP Range: ", round(max_range, 4))

# vielleicht nicht ganz so optimal? ist über alle features gemittelt
# average_range = feature_ranges.mean(axis=0)
#average_range_points = feature_ranges.mean(axis=1)

# Variability
explanation_var = shap_values_all.var(axis=0)

#Plots
for feat_idx in range(feature_ranges.shape[1]):
    plt.scatter(final_rate, feature_ranges[:, feat_idx], alpha=0.5)
    plt.xlim(0, 0.5)
    plt.ylim(0, max_range)
    plt.xlabel("Conflict Rate")
    plt.ylabel("SHAP Explanation Range")
    plt.title(f"{feature_names[feat_idx]} | Num. Models: {len(rashomon_models)} | "
              f"Accuracy: {acc.round(4)} | Num. conflicting Points: {len(conflict_data)}")
    plt.show()

"""
plt.scatter(final_rate, explanation_var, alpha=0.5)
plt.xlabel("Conflict Rate")
plt.ylabel("SHAP Explanation variability")
plt.title("VARIABILITY: Num. Models: " + str(len(rashomon_models))
          + ", Accuracy: " + str(acc.round(4))
          + ", Num. Points: " + str(len(conflict_data)))
plt.show()
"""
# TODO Vorzeichenwechsel von SHAP Werten
"""
Beobachtungen: 
- immer unterschiedliche Modelle
- Explanation Variability ist extrem klein
- Keine Korrelation bisher
- Laptop deutlich schneller
- Hey
"""

