
import json
import random
import math


def calculate_distance(city1, city2):
   
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def calculate_total_distance(tour, cities):
   
    total = 0
    for i in range(len(tour)):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % len(tour)]]
        total += calculate_distance(city1, city2)
    return total


def generate_random_cities(num_cities, width=800, height=600, seed=None):
   
    if seed is not None:
        random.seed(seed)
    
    cities = {}
    for i in range(num_cities):
        x = random.uniform(50, width - 50)
        y = random.uniform(50, height - 50)
        cities[i] = (x, y)
    
    return cities


def load_cities_from_file(filename):
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cities = {}
        for city_id, coords in data.items():
            cities[int(city_id)] = tuple(coords)
        
        return cities
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None


def save_cities_to_file(cities, filename):
   
    try:
        data = {str(k): list(v) for k, v in cities.items()}
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")
        return False


def get_neighbors(tour):
    
    neighbors = []
    for i in range(len(tour)):
        for j in range(i + 1, len(tour)):
            neighbor = tour.copy()
            # Đảo ngược đoạn từ i đến j (2-opt move)
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
            neighbors.append(neighbor)
    return neighbors


def create_distance_matrix(cities):
  
    distance_matrix = {}
    city_ids = list(cities.keys())
    
    for i in city_ids:
        for j in city_ids:
            if i != j:
                distance_matrix[(i, j)] = calculate_distance(cities[i], cities[j])
            else:
                distance_matrix[(i, j)] = 0
    
    return distance_matrix
