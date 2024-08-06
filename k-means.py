import numpy as np
import matplotlib.pyplot as plt

# Provided dataset
X = np.array([[3,2], [1,3], [3,2], [4,5], [2,3], [7,5], [6,4], [9,3], [8,3], [8,1]])

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_distances(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, x in enumerate(X):
        for j, centroid in enumerate(centroids):
            distances[i, j] = euclidean_distance(x, centroid)
    return np.round(distances, 4)

def assign_groups(distances):
    min_indices = np.argmin(distances, axis=1)
    group_matrix = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        min_index = min_indices[i]
        group_matrix[i, min_index] = 1
    return group_matrix

def print_iteration_info(iteration, X, distances, group_matrix, centroids):
    print(f"Iteration {iteration}:")
    print("Dataset Matrix:")
    print(X)
    print("Distance Matrix:")
    print(distances)
    print("Group Matrix:")
    print(group_matrix)
    print("Centroids:")
    print(np.round(centroids, 4))
    print()

# Plot the original dataset
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Original Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Choose the first data points as initial centroids
centroids = X[:2]

# Print the dataset matrix
print("Dataset Matrix:")
print(X)
print()

# Run KMeans clustering until convergence
for i in range(3):  # Change the number of iterations as needed
    # Calculate distances between each datapoint and centroids
    distances = calculate_distances(X, centroids)
    
    # Assign groups based on minimum distances
    group_matrix = assign_groups(distances)
    
    # Update centroids based on group assignments
    centroids = np.array([np.mean(X[group_matrix[:, k] == 1], axis=0) for k in range(centroids.shape[0])])
    
    # Print iteration information
    print_iteration_info(i + 1, X, distances, group_matrix, centroids)
