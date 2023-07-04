import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(n_samples, n_classes, noise=0.1):
    """
    Generuje zbiór danych "spiral" o zadanej liczbie próbek, klas i szumie.
    """
    X = np.zeros((n_samples*n_classes, 2))
    y = np.zeros(n_samples*n_classes, dtype='int')

    for class_idx in range(n_classes):
        start_angle = class_idx * 2 * np.pi / n_classes
        end_angle = (class_idx + 2) * 2 * np.pi / n_classes
        angles = np.linspace(start_angle, end_angle, n_samples)
        radii = np.linspace(0.1, 2, n_samples)
        X[class_idx*n_samples:(class_idx+1)*n_samples] = np.c_[radii*np.cos(angles), radii*np.sin(angles)]
        y[class_idx*n_samples:(class_idx+1)*n_samples] = class_idx

    # Dodanie szumu do danych
    X += np.random.randn(n_samples*n_classes, 2) * noise

    return X, y

def rotate_spiral(X, angle):
    """
    Obraca spiralę o zadany kąt.
    """
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_X = X.dot(rotation_matrix)
    return rotated_X

n_samples = 5000
n_classes = 2
noise = 0.1

n_samples = 500
n_classes = 2

X_spiral, y_spiral = generate_spiral(n_samples, n_classes)

plt.figure(figsize=(6, 6))
plt.scatter(X_spiral[:, 0], X_spiral[:, 1], c=y_spiral, cmap='viridis')
plt.title('Spiral')
plt.show()
