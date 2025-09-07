import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
# We no longer import or use the outdated algorithm_globals

# --- Task 1: Create a dataset ---
num_samples = 20
angel_numbers = [i * 111 for i in range(1, 10)] * (num_samples // 9 + 1)
angel_numbers = angel_numbers[:num_samples]

regular_numbers = []
while len(regular_numbers) < num_samples:
    num = np.random.randint(100, 1000)
    digits = [int(d) for d in str(num)]
    if len(set(digits)) > 1:
        regular_numbers.append(num)

dataset_numbers = angel_numbers + regular_numbers
print(f"Generated {len(dataset_numbers)} numbers for the dataset.")

# --- Task 2: Engineer Features ---
def number_to_features(n):
    digits = [int(d) for d in str(n)]
    first_digit = digits[0]
    std_dev = np.std(digits)
    return [first_digit, std_dev]

features = np.array([number_to_features(n) for n in dataset_numbers])
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# --- Task 3: Create Labels ---
labels = np.array([1] * num_samples + [0] * num_samples)

# --- Task 4: Build a QML Model ---
num_features = 2
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
# The modern way to ensure reproducibility is to pass the seed directly here
sampler = Sampler(options={"seed": 42})
vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz)

# --- Task 5: Train and Score ---
print("\nTraining the Quantum Machine Learning model on the number dataset...")
vqc.fit(features_scaled, labels)
print("Training complete!")
score = vqc.score(features_scaled, labels)
print(f"Training accuracy: {score * 100:.2f}%")

# --- Task 6: Visualize ---
print("\nCreating visualization...")
plt.figure(figsize=(8, 8))
plt.title("Angel Number QML Classifier")
plt.xlabel("Feature 1: First Digit (Normalized)")
plt.ylabel("Feature 2: Standard Deviation of Digits (Normalized)")

grid_resolution = 50
grid_x = np.linspace(0, 1, grid_resolution)
grid_y = np.linspace(0, 1, grid_resolution)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_points = np.vstack([grid_xx.ravel(), grid_yy.ravel()]).T
grid_predictions = vqc.predict(grid_points)
grid_zz = grid_predictions.reshape(grid_xx.shape)
plt.contourf(grid_xx, grid_yy, grid_zz, alpha=0.3)

angel_features = features_scaled[:num_samples]
regular_features = features_scaled[num_samples:]
plt.scatter(angel_features[:, 0], angel_features[:, 1], color='cyan', marker='*', s=100, label='Angel Numbers (Class 1)')
plt.scatter(regular_features[:, 0], regular_features[:, 1], color='purple', marker='o', label='Regular Numbers (Class 0)')

plt.legend()
plt.grid(True)
plt.show()