import numpy as np
import matplotlib.pyplot as plt
import shap

# Synthetic Data

rng = np.random.default_rng(11)
std = 0.1

# class 1
X1 = rng.normal(loc=(0.5, 0.5), scale=std, size=(100, 2))

# class 2
X2 = rng.normal(loc=(1.5, 1.5), scale=std, size=(100, 2))

all_points = np.concatenate((X1, X2))

# Plot
plt.scatter(X1[:,0], X1[:,1], label="class 1")
plt.scatter(X2[:,0], X2[:,1], label="class 2")
plt.plot([0.0, 2.0], [2.0, 0.0], label="Classifier 1", color="green")
plt.axvline(x=1.0, label="Classifier 2", color="black")
plt.axhline(y=1.0, label="Classifier 3", color="orange")
plt.plot([0.0, 2.0], [1.5, 0.5], label="Classifier 4", color="blue")
plt.plot([0.0, 2.0], [1.5, 0.0], label="Classifier 5", color="red")
plt.xlim(left=0.0, right=2.0)
plt.ylim(bottom=0.0, top=2.0)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.xlabel("x0")
plt.ylabel("x1")
plt.tight_layout()
plt.show()

# self-defined linear classifiers
class LinClassifier:
    def __init__(self, w: np.ndarray, b: float):
        self.w = np.array(w, dtype=float)
        self.b = float(b)

    def getName(self):
        if self.w[1]==0:
            b = -self.b/self.w[0]
            return "x = " + str(b)
        a = -self.w[0]/self.w[1]
        b = -self.b / self.w[1]
        return "y = " + str(a) + "x + " + str(b)

    def predict(self, coordinates: np.ndarray) -> int:
        x, y = coordinates
        val = self.w[0] * x + self.w[1] * y + self.b
        return int(np.sign(val))

    def predict_multiple(self, points: np.ndarray) -> np.ndarray:
        result = np.zeros(points.shape[0])
        for index, point in enumerate(points):
            result[index] = self.predict(point)
        return result

# classifier examples
classifiers = [
    LinClassifier(w=[1, 1], b=-2), #half
    LinClassifier(w=[1,0], b=-1), #vertical
    LinClassifier(w=[0,1], b=-1), #horizontal
    LinClassifier(w=[0.5,1], b=-1.5),
    LinClassifier(w=[0.75,1], b=-1.5)
]

# test points
point_1 = [0.5,0.5]
point_2 = [1.5,1.5]

# tests
"""for classifier in classifiers:
    print(f" Point {point_1}: {classifier.predict(point_1)}")
    print(f" Point {point_2}: {classifier.predict(point_2)}")

for classifier in classifiers:
    print(f" Should be all -1: {classifier.predict_multiple(X1)}")
    print(f" Should be all 1: {classifier.predict_multiple(X2)}")
"""

# ambiguity -> Marx, Calmon, Ustun "Predictive Multiplicity in Classification"
def ambiguity(classifiers, points):
    h_0 = classifiers[0]
    others = classifiers[1:]
    n = len(points)
    counter_disagree = 0
    for point in points:
        pred_h_0 = h_0.predict(point)
        disagree = False

        for classifier in others:
            if classifier.predict(point) != pred_h_0:
                disagree = True
                #print(classifier.b, point)
        if disagree:
            counter_disagree += 1
    return counter_disagree / n

print(ambiguity(classifiers, all_points))

# accuracy for comparison if adjusting std
def accuracy(classifiers, points):
    accuracy_dict = {}
    for classifier in classifiers:
        wrong = 0
        for point in points:
            if (classifier.predict(point)) == 1 and any(np.array_equal(point, row) for row in X2):
                pass
            elif (classifier.predict(point) == -1) and any(np.array_equal(point, row) for row in X1):
                pass
            else:
                wrong += 1
        acc = 1 - (wrong / len(points))
        print(acc)
        accuracy_dict[classifier.getName()] = acc
    return accuracy_dict

#print(accuracy(classifiers, all_points))

# SHAP

# wrapping classifier in vectorized function
def predict_fn(X):
    X = np.asarray(X)
    return clf.predict_multiple(X).astype(float)

sample = shap.sample(all_points, 50)


for classifier in classifiers:
    clf = classifier
    explainer = shap.KernelExplainer(predict_fn, sample)
    shap_values = explainer.shap_values(all_points)
    print(classifier.getName())
    shap.summary_plot(shap_values, all_points, feature_names=["x0", "x1"])
