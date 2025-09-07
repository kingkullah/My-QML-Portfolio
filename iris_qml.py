import numpy as np
import matplotlib.pyplot as plt

# We'll use scikit-learn to load the real-world Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC

print("Loading real-world Iris flower dataset...")

# --- 1. Load and Prepare the Real-World Data ---
iris_data = load_iris()

# We will only use two features: petal length (index 2) and petal width (index 3)
features = iris_data.data[:, [2, 3]]
# We will only try to classify one species (Setosa, label 0) vs. the others (labels 1 and 2)
labels = (iris_data.target != 0) * 1  # Setosa becomes 0, everything else becomes 1

# Normalize the features to be between 0 and 1, just like before
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into a training set and a testing set
# The model learns on the training set and we test its skill on the unseen testing set
train_features, test_features, train_labels, test_labels = train_test_split(
    features_scaled, labels, train_size=0.75, random_state=42
)

# --- 2. Build the same QML Model as before ---
num_features = 2
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
sampler = Sampler(options={"seed": 42})
vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz)

# --- 3. Train the model on the real flower data ---
print("\nTraining the QML model on Iris flower data...")
vqc.fit(train_features, train_labels)
print("Training complete!")

# --- 4. Score the model on the unseen test data ---
train_score = vqc.score(train_features, train_labels)
test_score = vqc.score(test_features, test_labels)
print(f"\nTraining accuracy: {train_score * 100:.2f}%")
print(f"Testing accuracy on unseen data: {test_score * 100:.2f}%")

# --- 5. Visualize the Results ---
print("\nCreating visualization...")
plt.figure(figsize=(8, 8))
plt.title("Iris Flower QML Classifier")
plt.xlabel("Petal Length (Normalized)")
plt.ylabel("Petal Width (Normalized)")

# Plot the decision boundary
grid_resolution = 50
grid_x = np.linspace(0, 1, grid_resolution)
grid_y = np.linspace(0, 1, grid_resolution)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T
grid_predictions = vqc.predict(grid_points)
grid_zz = grid_predictions.reshape(grid_xx.shape)
plt.contourf(grid_xx, grid_yy, grid_zz, alpha=0.3)

# Plot the data points
setosa_features = train_features[train_labels == 0]
other_features = train_features[train_labels == 1]
plt.scatter(setosa_features[:, 0], setosa_features[:, 1], color='blue', marker='o', label='Setosa (Class 0)')
plt.scatter(other_features[:, 0], other_features[:, 1], color='red', marker='s', label='Other Species (Class 1)')

plt.legend()
plt.grid(True)
plt.show()