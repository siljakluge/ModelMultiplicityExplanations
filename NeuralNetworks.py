import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

X_df, y = load_breast_cancer(return_X_y=True, as_frame=True) #https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
y = y.astype(int).values  # 0 = benign, 1 = malignant
X = X_df.copy()

# Scaling
num_cols = X.columns.tolist()
pre = ColumnTransformer([("num", StandardScaler(), num_cols)], remainder="drop")

def create_mlp():
    classifier = MLPClassifier(hidden_layer_sizes=(32,16), early_stopping=True)
    return Pipeline([("pre", pre), ("clf", classifier)])

def rashomon_set(X, y, n_models=50, tolerance=0.01, base_seed=11):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=11)
    rng = np.random.RandomState(base_seed)
    random_seeds = rng.randint(0, 10_000, size=n_models)

    models, accuracies = [], []
    for seed in random_seeds:
        pipe = create_mlp()
        pipe.set_params(clf__random_state=seed)
        pipe.fit(X_tr, y_tr)

        acc_score = accuracy_score(y_val, pipe.predict(X_val))
        models.append(pipe)
        accuracies.append(acc_score)

    accs = np.array(accuracies)
    best_acc = accs.max()
    threshold = best_acc - tolerance
    keep = accs >= threshold
    good_models = [m for m, k in zip(models, keep) if k]

    return good_models, (X_val, y_val), best_acc


def ambiguity(classifiers, points):
    if len(classifiers) < 2:
        return 0.0
    predictions = np.vstack([m.predict(points) for m in classifiers])
    h_0 = predictions[0]
    disagrees = np.any(predictions != h_0, axis=0)
    return float(disagrees.mean())

# Test
rashomon_models, (X_val, y_val), acc = rashomon_set(X, y, n_models=100, tolerance=0.015, base_seed=11)
print("Models:", len(rashomon_models))
print("Best Accuracy:", acc.round(4))
print("Ambiguity:", round(ambiguity(rashomon_models, X_val), 4))


"""

#default
hidden_layer_sizes=(100,):
Models: 1
Best Accuracy 0.9532
Ambiguity: 0.0

For hidden_layer_sizes=(16,8): 
Models: 2
Best Accuracy 0.9474
Ambiguity: 0.076

hidden_layer_sizes=(10,5):
Models: 2
Best Accuracy 0.9298
Ambiguity: 0.0526

hidden_layer_sizes=(10,8):
Models: 4
Best Accuracy 0.9415
Ambiguity: 0.0936

hidden_layer_sizes=(32,16):
Models: 10
Best Accuracy 0.9415
Ambiguity: 0.1053

hidden_layer_sizes=(100,8):
Models: 2
Best Accuracy 0.9649
Ambiguity: 0.0175
"""

#TODO Training acc