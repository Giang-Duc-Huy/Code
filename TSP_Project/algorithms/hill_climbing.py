
import random
import time
from typing import List, Tuple, Optional


class HillClimbing:
    def __init__(self, tsp_problem, max_iterations=10000, max_restarts=10):
       
        if tsp_problem is None:
            raise ValueError("tsp_problem không được None")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations phải > 0, nhận được: {max_iterations}")
        if max_restarts <= 0:
            raise ValueError(f"max_restarts phải > 0, nhận được: {max_restarts}")
        
        self.tsp = tsp_problem
        self.max_iterations = max_iterations
        self.max_restarts = max_restarts
        self.best_route = None
        self.best_distance = float('inf')
        self.history = []  # Lưu lịch sử để vẽ đồ thị
        self.execution_time = 0
    
    def _two_opt_swap(self, route: List[int], i: int, k: int) -> List[int]:
      
        new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
        return new_route
    
    def _best_2opt_neighbor(self, route: List[int]) -> Tuple[List[int], float, bool]:
      
        best_distance = self.tsp.calculate_route_distance(route)
        best_route = route.copy()
        improved = False
        n = len(route)
        
        for i in range(n - 1):
            for k in range(i + 1, n):
                neighbor = self._two_opt_swap(route, i, k)
                distance = self.tsp.calculate_route_distance(neighbor)
                if distance < best_distance:
                    best_distance = distance
                    best_route = neighbor
                    improved = True
        
        return best_route, best_distance, improved
    
    def solve(self, callback=None, verbose=True):
     
        start_time = time.time()
        
        overall_best_route = None
        overall_best_distance = float('inf')
        self.history = []
        history_counter = 0
        
        if verbose:
            print("=" * 60)
            print("THUẬT TOÁN HILL CLIMBING")
            print("=" * 60)
            print(f"Số thành phố: {self.tsp.num_cities}")
            print(f"Số lần restart: {self.max_restarts}")
        
        for restart in range(self.max_restarts):
            # Khởi tạo nghiệm ngẫu nhiên
            current_route = self.tsp.generate_random_route()
            current_distance = self.tsp.calculate_route_distance(current_route)
            
            if verbose:
                print(f"\nRestart #{restart + 1}: Khoảng cách ban đầu = {current_distance:.2f}")
            
            iteration = 0
            local_history = [(0, current_distance)]
            
            # Leo đồi cho đến khi không cải thiện được nữa
            while iteration < self.max_iterations:
                neighbor_route, neighbor_distance, improved = self._best_2opt_neighbor(current_route)
                iteration += 1
                
                if improved:
                    current_route = neighbor_route
                    current_distance = neighbor_distance
                    local_history.append((iteration, current_distance))
                    
                    # Gọi callback mỗi 10 iterations
                    if callback and iteration % 10 == 0:
                        callback(current_route, current_distance, history_counter + iteration)
                    
                    if verbose and iteration % 100 == 0:
                        print(f"  Iteration {iteration}: Khoảng cách = {current_distance:.2f}")
                else:
                    # Không cải thiện được nữa -> đạt local optimum
                    break
            
            if verbose:
                print(f"  Kết thúc sau {iteration} iterations: {current_distance:.2f}")
            
            # Cập nhật best overall
            if current_distance < overall_best_distance:
                overall_best_route = current_route.copy()
                overall_best_distance = current_distance
            
            # Merge history
            offset = history_counter
            for it, dist in local_history:
                self.history.append({
                    'iteration': offset + it,
                    'distance': dist,
                    'best_distance': overall_best_distance
                })
            history_counter += iteration
        
        self.best_route = overall_best_route
        self.best_distance = overall_best_distance
        self.execution_time = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"KẾT QUẢ CUỐI CÙNG")
            print(f"Khoảng cách tốt nhất: {self.best_distance:.2f}")
            print(f"Thời gian thực thi: {self.execution_time:.3f} giây")
            print("=" * 60)
        
        return {
            'tour': self.best_route,
            'distance': self.best_distance,
            'time': self.execution_time,
            'iterations': history_counter,
            'restarts': self.max_restarts,
            'history': self.history
        }
    
    def get_solution(self):
     
        return self.best_route, self.best_distance
    
    def get_algorithm_info(self) -> dict:
        """Lấy thông tin về thuật toán"""
        return {
            'name': 'Hill Climbing',
            'max_iterations': self.max_iterations,
            'max_restarts': self.max_restarts
        }
