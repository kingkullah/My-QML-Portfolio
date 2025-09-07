import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.utils import algorithm_globals

# --- Task 0: Setup ---
# Ensure our results are reproducible
algorithm_globals.random_seed = 42
num_samples = 20 # Number of samples for each class

# --- Task 1: Create a dataset ---
# Generate 20 "angel numbers"
angel_numbers = [i * 111 for i in range(1, 10)] * (num_samples // 9 + 1)
angel_numbers = angel_numbers[:num_samples]

# Generate 20 random 3-digit "regular numbers"
regular_numbers = []
while len(regular_numbers) < num_samples:
    num = np.random.randint(100, 1000)
    # Ensure it's not an angel number by checking if all digits are the same
    digits = [int(d) for d in str(num)]
    if len(set(digits)) > 1:
        regular_numbers.append(num)

dataset_numbers = angel_numbers + regular_numbers
print(f"Generated {len(dataset_numbers)} numbers for the dataset.")

# --- Task 2: Engineer Features ---
# This function will extract features from a single number
def number_to_features(n):
    digits = [int(d) for d in str(n)]
    first_digit = digits[0]
    std_dev = np.std(digits)
    return [first_digit, std_dev]

# Calculate features for all our numbers
features = np.array([number_to_features(n) for n in dataset_numbers])

# Important: Normalize all features so they are scaled between 0.0 and 1.0
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# --- Task 3: Create Labels ---
# Assign a label of 1 to the angel numbers and 0 to the regular numbers.
labels = np.array([1] * num_samples + [0] * num_samples)

# --- Task 4: Build a QML Model ---
num_features = 2
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
sampler = Sampler(options={"seed": algorithm_globals.random_seed})
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
angel_features = features_scaled[:num_samples]
regular_features = features_scaled[num_samples:]
plt.scatter(angel_features[:, 0], angel_features[:, 1], color='cyan', marker='*', s=100, label='Angel Numbers (Class 1)')
plt.scatter(regular_features[:, 0], regular_features[:, 1], color='purple', marker='o', label='Regular Numbers (Class 0)')

plt.legend()
plt.grid(True)
plt.show()