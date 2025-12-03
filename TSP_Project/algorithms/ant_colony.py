
import random
import time
import numpy as np


class Ant:
    
    def __init__(self, num_cities):
     
        self.num_cities = num_cities
        self.tour = []  # Hành trình của kiến
        self.distance = float('inf')  # Tổng khoảng cách
        self.visited = set()  # Các thành phố đã thăm
    
    def reset(self):
        """Reset trạng thái của kiến"""
        self.tour = []
        self.distance = float('inf')
        self.visited = set()
    
    def visit_city(self, city):
        
        self.tour.append(city)
        self.visited.add(city)
    
    def can_visit(self, city):
       
        return city not in self.visited
    
    def get_unvisited_cities(self, all_cities):
        
        return [city for city in all_cities if city not in self.visited]


class AntColonyOptimization:
    def __init__(self, tsp_problem, num_ants=50, num_iterations=100, 
                 alpha=1.0, beta=5.0, evaporation=0.5, q=100):
        
        # Validate parameters
        if tsp_problem is None:
            raise ValueError("tsp_problem không được None")
        if num_ants < 2:
            raise ValueError(f"Số kiến phải >= 2, nhận được: {num_ants}")
        if num_ants > 200:
            raise ValueError(f"Số kiến quá lớn (> 200), nhận được: {num_ants}")
        if num_iterations <= 0:
            raise ValueError(f"Số iterations phải > 0, nhận được: {num_iterations}")
        if alpha < 0:
            raise ValueError(f"Alpha phải >= 0, nhận được: {alpha}")
        if beta < 0:
            raise ValueError(f"Beta phải >= 0, nhận được: {beta}")
        if not (0 <= evaporation <= 1):
            raise ValueError(f"Evaporation phải trong [0, 1], nhận được: {evaporation}")
        if q <= 0:
            raise ValueError(f"Q phải > 0, nhận được: {q}")
        
        self.tsp = tsp_problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        
        self.num_cities = tsp_problem.num_cities
        self.city_ids = list(range(self.num_cities))
        
        # Khởi tạo đàn kiến
        self.ants = [Ant(self.num_cities) for _ in range(self.num_ants)]
        
        # Khởi tạo ma trận pheromone và heuristic
        self.pheromone = {}
        self.heuristic = {}
        
        self._initialize_matrices()
        
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
        self.execution_time = 0
    
    def _initialize_matrices(self):
        """
        Khởi tạo ma trận pheromone và heuristic
        """
        initial_pheromone = 1.0
        
        for i in self.city_ids:
            for j in self.city_ids:
                if i != j:
                    dist = self.tsp.distance_matrix[i][j]
                    self.pheromone[(i, j)] = initial_pheromone
                    # Heuristic = 1/distance (thành phố càng gần càng hấp dẫn)
                    self.heuristic[(i, j)] = 1.0 / dist if dist > 0 else 0
                else:
                    self.pheromone[(i, j)] = 0
                    self.heuristic[(i, j)] = 0
    
    def _select_next_city(self, current_city, unvisited):
       
        if not unvisited:
            return None
        
        # Tính xác suất cho mỗi thành phố chưa thăm
        probabilities = []
        total = 0
        
        for city in unvisited:
            pheromone_val = self.pheromone[(current_city, city)] ** self.alpha
            heuristic_val = self.heuristic[(current_city, city)] ** self.beta
            prob = pheromone_val * heuristic_val
            probabilities.append(prob)
            total += prob
        
        # Normalize probabilities
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Nếu tất cả xác suất = 0, chọn ngẫu nhiên
            probabilities = [1.0 / len(unvisited)] * len(unvisited)
        
        # Chọn thành phố dựa trên xác suất
        selected_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[selected_idx]
    
    def _construct_solution(self, ant):
       
        # Reset trạng thái kiến
        ant.reset()
        
        # Chọn thành phố bắt đầu ngẫu nhiên
        start_city = random.choice(self.city_ids)
        ant.visit_city(start_city)
        
        current_city = start_city
        
        # Xây dựng hành trình
        while len(ant.tour) < self.num_cities:
            unvisited = ant.get_unvisited_cities(self.city_ids)
            if not unvisited:
                break
            
            next_city = self._select_next_city(current_city, unvisited)
            ant.visit_city(next_city)
            current_city = next_city
        
        # Tính khoảng cách của hành trình
        ant.distance = self.tsp.calculate_route_distance(ant.tour)
    
    def _update_pheromone(self, all_tours, all_distances):
       
        # Bay hơi pheromone (evaporation)
        for i in self.city_ids:
            for j in self.city_ids:
                if i != j:
                    self.pheromone[(i, j)] *= (1 - self.evaporation)
                    # Đảm bảo pheromone không âm và có giá trị tối thiểu
                    if self.pheromone[(i, j)] < 0.0001:
                        self.pheromone[(i, j)] = 0.0001
        
        # Thêm pheromone mới từ các con kiến
        for tour, distance in zip(all_tours, all_distances):
            # Lượng pheromone deposit tỷ lệ nghịch với khoảng cách
            pheromone_deposit = self.q / distance if distance > 0 else 0
            
            for i in range(len(tour)):
                city1 = tour[i]
                city2 = tour[(i + 1) % len(tour)]
                self.pheromone[(city1, city2)] += pheromone_deposit
                self.pheromone[(city2, city1)] += pheromone_deposit
    
    def solve(self, callback=None, verbose=True):
        
        start_time = time.time()
        
        self.best_tour = None
        self.best_distance = float('inf')
        self.history = []
        
        if verbose:
            print("=" * 60)
            print("THUẬT TOÁN ANT COLONY OPTIMIZATION")
            print("=" * 60)
            print(f"Số thành phố: {self.num_cities}")
            print(f"Số kiến: {self.num_ants}")
            print(f"Số thế hệ: {self.num_iterations}")
            print(f"Alpha (pheromone): {self.alpha}")
            print(f"Beta (heuristic): {self.beta}")
            print(f"Evaporation rate: {self.evaporation}")
        
        for iteration in range(self.num_iterations):
            # Các con kiến xây dựng nghiệm
            all_tours = []
            all_distances = []
            
            for ant in self.ants:
                self._construct_solution(ant)
                all_tours.append(ant.tour)
                all_distances.append(ant.distance)
                
                # Cập nhật nghiệm tốt nhất
                if ant.distance < self.best_distance:
                    self.best_distance = ant.distance
                    self.best_tour = ant.tour.copy()
            
            # Cập nhật pheromone
            self._update_pheromone(all_tours, all_distances)
            
            # Lưu lịch sử
            avg_distance = sum(all_distances) / len(all_distances)
            self.history.append({
                'iteration': iteration,
                'best_distance': self.best_distance,
                'avg_distance': avg_distance
            })
            
            # Gọi callback
            if callback and (iteration + 1) % 10 == 0:
                callback(self.best_tour, self.best_distance, iteration + 1)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Thế hệ {iteration + 1}/{self.num_iterations}: "
                      f"Best = {self.best_distance:.2f}, Avg = {avg_distance:.2f}")
        
        self.execution_time = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"KẾT QUẢ CUỐI CÙNG")
            print(f"Khoảng cách tốt nhất: {self.best_distance:.2f}")
            print(f"Thời gian thực thi: {self.execution_time:.3f} giây")
            print("=" * 60)
        
        return {
            'tour': self.best_tour,
            'distance': self.best_distance,
            'time': self.execution_time,
            'iterations': self.num_iterations,
            'history': self.history
        }
    
    def get_solution(self):
       
        return self.best_tour, self.best_distance
    
    def get_algorithm_info(self):
        """Lấy thông tin về thuật toán"""
        return {
            'name': 'Ant Colony Optimization (ACO)',
            'num_ants': self.num_ants,
            'num_iterations': self.num_iterations,
            'alpha': self.alpha,
            'beta': self.beta,
            'evaporation': self.evaporation
        }
