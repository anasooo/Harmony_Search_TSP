import random

import matplotlib.pyplot as plt
import numpy as np


# Define the objective function (total distance in TSP)
def total_distance(path, distance_matrix):
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += distance_matrix[path[i]][path[i + 1]]
    total_dist += distance_matrix[path[-1]][path[0]]
    return total_dist

def generate_harmony(n_cities):
    path = random.sample(range(1, n_cities), n_cities - 1)
    path.append(path[0])
    return path


# Function to read cities from a file
def read_cities_from_file(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()

  for i, line in enumerate(lines):
    if line.strip() == 'NODE_COORD_SECTION':
      start_index = i + 1
      break
  #read the city coordinates
  coords = []
  for line in lines[start_index:]:
    if line.strip() == 'EOF':
      break
    values = line.strip().split()
    if len(values) < 3:
      continue
    x, y = values[1:3]
    coords.append((float(x), float(y)))
  return coords

# Function to generate distance matrix from cities
def generate_distance_matrix(cities):
    n_cities = len(cities)
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            distance = np.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

# Harmony Search Algorithm
def harmony_search_tsp(distance_matrix, n_harmonies, max_iter):
    n_cities = len(distance_matrix)
    harmonies = [generate_harmony(n_cities) for _ in range(n_harmonies)]

    for _ in range(max_iter):
        new_harmony = generate_harmony(n_cities)
        min_harmony = min(harmonies, key=lambda x: total_distance(x, distance_matrix))
        if total_distance(new_harmony, distance_matrix) < total_distance(min_harmony, distance_matrix):
            harmonies.remove(min_harmony)
            harmonies.append(new_harmony)

    best_solution = min(harmonies, key=lambda x: total_distance(x, distance_matrix))
    best_distance = total_distance(best_solution, distance_matrix)
    return best_solution, best_distance

# Example usage
filename = "berlin52.tsp"  # Replace "cities.txt" with your file containing city coordinates
cities = read_cities_from_file(filename)
distance_matrix = generate_distance_matrix(cities)

n_harmonies = 100
max_iter = 800

best_path, best_distance = harmony_search_tsp(distance_matrix, n_harmonies, max_iter)
print("Best Path:", best_path)
print("Total Distance:", best_distance)

# Visualization
def plot_tsp_solution(cities, path):
    x = [city[0] for city in cities]
    y = [city[1] for city in cities]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='red')
    for i in range(len(path) - 1):
        plt.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]], color='blue')
    plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], color='blue')  # connect last city to first city
    plt.title('TSP Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

plot_tsp_solution(cities, best_path)