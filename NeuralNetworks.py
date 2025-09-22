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

datapoint_subset = X_val[:1000]
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
    shap_values_all.append(shap_values[1])

shap_values_all = np.array(shap_values_all)

# Varaibility
explanation_var = shap_values_all.var(axis=0).mean(axis=1)

#Plot
plt.scatter(final_rate, explanation_var, alpha=0.5)
plt.xlabel("Conflict Rate")
plt.ylabel("SHAP Explanation variability")
plt.title("Number of Models: " + str(len(rashomon_models))
          + ", Best Accuracy: " + str(acc.round(4))
          + ", Number of conflicting Points: " + str(len(conflict_data)))
plt.show()

"""
Beobachtungen: 
- immer unterschiedliche Modelle
- Explanation Variability ist extrem klein
- Keine Korrelation bisher
"""

