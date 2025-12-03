"""
Module định nghĩa bài toán TSP (Travelling Salesman Problem)
"""
import numpy as np
import random


class TSProblem:
    """Lớp đại diện cho bài toán người du lịch"""
    
    def __init__(self, num_cities=20, city_coords=None):
        # Validate num_cities
        if num_cities < 2:
            raise ValueError(
                f"Số thành phố phải >= 2, nhận được: {num_cities}\n\n"
                f"Giải thích: Cần ít nhất 2 thành phố để tạo thành một tuyến đường."
            )
        if num_cities == 2:
            raise ValueError(
                f"Số thành phố = 2 không phù hợp\n\n"
                f"Giải thích: Với 2 thành phố, chỉ có 1 cách duy nhất (A→B→A).\n"
                f"Không cần thuật toán tối ưu hóa.\n\n"
                f"Khuyến nghị: Sử dụng >= 3 thành phố."
            )
        if num_cities > 500:
            raise ValueError(
                f"Số thành phố quá lớn (> 500), nhận được: {num_cities}\n\n"
                f"Giải thích: Số thành phố quá lớn sẽ:\n"
                f"- Làm tăng thời gian tính toán rất nhiều\n"
                f"- Tiêu tốn quá nhiều bộ nhớ\n"
                f"- Có thể làm treo hoặc crash ứng dụng\n\n"
                f"Khuyến nghị: Sử dụng <= 200 thành phố để đảm bảo hiệu suất tốt."
            )
        
        self.num_cities = num_cities
        
        if city_coords is not None:
            city_coords_array = np.array(city_coords)
            # Validate city_coords
            if len(city_coords_array.shape) != 2 or city_coords_array.shape[1] != 2:
                raise ValueError("Tọa độ thành phố phải là mảng 2 chiều với shape (n, 2)")
            if len(city_coords_array) < 3:
                raise ValueError(f"Số lượng tọa độ phải >= 3, nhận được: {len(city_coords_array)}")
            
            self.city_coords = city_coords_array
            self.num_cities = len(city_coords)
        else:
            # Tạo ngẫu nhiên tọa độ các thành phố trong khoảng [0, 100]
            self.city_coords = np.random.rand(num_cities, 2) * 100
        
        # Tính ma trận khoảng cách
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self):
        """Tính ma trận khoảng cách giữa các thành phố"""
        n = self.num_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (self.city_coords[i][0] - self.city_coords[j][0]) ** 2 +
                    (self.city_coords[i][1] - self.city_coords[j][1]) ** 2
                )
                dist_matrix[i][j] = dist
                dist_matrix[j][i] = dist
        
        return dist_matrix
    
    def calculate_route_distance(self, route):
       
        # Validate route
        if not route or len(route) == 0:
            raise ValueError("Tuyến đường không được rỗng")
        if len(route) != self.num_cities:
            raise ValueError(f"Độ dài tuyến đường ({len(route)}) phải bằng số thành phố ({self.num_cities})")
        if len(set(route)) != len(route):
            raise ValueError("Tuyến đường chứa thành phố trùng lặp")
        if min(route) < 0 or max(route) >= self.num_cities:
            raise ValueError(f"Chỉ số thành phố phải trong khoảng [0, {self.num_cities-1}]")
        
        total_distance = 0
        for i in range(len(route)):
            city1 = route[i]
            city2 = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[city1][city2]
        return total_distance
    
    def generate_random_route(self):
        """Tạo một tuyến đường ngẫu nhiên"""
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route
    
    def get_city_coords(self):
        """Lấy tọa độ các thành phố"""
        return self.city_coords.copy()
