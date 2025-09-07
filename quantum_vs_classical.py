import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC # Import the classical Support Vector Classifier

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC

print("Loading real-world Iris flower dataset...")

# --- 1. Load and Prepare the Data (Same as before) ---
iris_data = load_iris()
features = iris_data.data[:, [2, 3]] # Petal length and width
labels = (iris_data.target != 0) * 1  # Setosa (0) vs. Others (1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
train_features, test_features, train_labels, test_labels = train_test_split(
    features_scaled, labels, train_size=0.75, random_state=42
)

# --- 2. Train the Quantum Model (Same as before) ---
print("\nTraining the Quantum Machine Learning model...")
num_features = 2
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
sampler = Sampler(options={"seed": 42})
vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz)
vqc.fit(train_features, train_labels)
print("Quantum Training complete!")

# --- 3. Train the Classical Model ---
print("\nTraining the Classical Machine Learning model...")
svm = SVC(kernel='rbf', gamma='auto') # Create a standard SVM classifier
svm.fit(train_features, train_labels)
print("Classical Training complete!")

# --- 4. Compare the Accuracy ---
vqc_test_score = vqc.score(test_features, test_labels)
svm_test_score = svm.score(test_features, test_labels)
print(f"\nQuantum Classifier Test Accuracy: {vqc_test_score * 100:.2f}%")
print(f"Classical SVM Test Accuracy:    {svm_test_score * 100:.2f}%")


# --- 5. Visualize Both Decision Boundaries ---
print("\nCreating side-by-side visualization...")
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --- Plot for VQC ---
ax1.set_title("Quantum Classifier (VQC) Decision Boundary")
ax1.set_xlabel("Petal Length (Normalized)")
ax1.set_ylabel("Petal Width (Normalized)")
# Create the grid
grid_resolution = 50
grid_x = np.linspace(0, 1, grid_resolution)
grid_y = np.linspace(0, 1, grid_resolution)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T
# Get predictions and plot
grid_predictions_vqc = vqc.predict(grid_points)
grid_zz_vqc = grid_predictions_vqc.reshape(grid_xx.shape)
ax1.contourf(grid_xx, grid_yy, grid_zz_vqc, alpha=0.3)
# Plot data points
setosa_features = train_features[train_labels == 0]
other_features = train_features[train_labels == 1]
ax1.scatter(setosa_features[:, 0], setosa_features[:, 1], color='blue', marker='o', label='Setosa (Class 0)')
ax1.scatter(other_features[:, 0], other_features[:, 1], color='red', marker='s', label='Other Species (Class 1)')
ax1.legend()
ax1.grid(True)

# --- Plot for SVM ---
ax2.set_title("Classical Classifier (SVM) Decision Boundary")
ax2.set_xlabel("Petal Length (Normalized)")
ax2.set_ylabel("Petal Width (Normalized)")
# Get predictions and plot
grid_predictions_svm = svm.predict(grid_points)
grid_zz_svm = grid_predictions_svm.reshape(grid_xx.shape)
ax2.contourf(grid_xx, grid_yy, grid_zz_svm, alpha=0.3)
# Plot data points
ax2.scatter(setosa_features[:, 0], setosa_features[:, 1], color='blue', marker='o', label='Setosa (Class 0)')
ax2.scatter(other_features[:, 0], other_features[:, 1], color='red', marker='s', label='Other Species (Class 1)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()