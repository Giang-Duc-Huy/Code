"""
TSP Solver - Hill Climbing với Random Restarts
Giải quyết bài toán Travelling Salesman Problem bằng thuật toán Hill Climbing

Cài đặt thư viện:
pip install PyQt6 matplotlib numpy

Cách chạy:
python tsp_solver.py
"""

import sys
import random
import math
import statistics
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QTableWidget, QTableWidgetItem, QSplitter)
from PyQt6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TSPSolver:
    """Lớp giải quyết bài toán TSP bằng Hill Climbing"""
    
    def __init__(self, campuses):
        self.campuses = campuses
        self.num_campuses = len(campuses)
        self.distance_matrix = self.calculate_distance_matrix()
        self.best_distance = float('inf')
        self.best_route = None
        self.history = []
    
    def calculate_distance_matrix(self):
        """Tạo ma trận khoảng cách giữa các campus"""
        n = self.num_campuses
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self.euclidean_distance(
                        self.campuses[i][1], 
                        self.campuses[j][1]
                    )
        return matrix
    
    def euclidean_distance(self, coord1, coord2):
        """Tính khoảng cách Euclid giữa 2 điểm"""
        return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    def total_distance(self, route):
        """Tính tổng khoảng cách của một tuyến đường"""
        total = 0
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        total += self.distance_matrix[route[-1]][route[0]]
        return total
    
    def get_neighbors(self, route):
        """Sinh tất cả các neighbor bằng cách swap 2 vị trí"""
        neighbors = []
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def hill_climbing(self, initial_route):
        """Thuật toán Steepest-Ascent Hill Climbing"""
        current_route = initial_route.copy()
        current_distance = self.total_distance(current_route)
        iterations = 0
        
        while True:
            neighbors = self.get_neighbors(current_route)
            best_neighbor = None
            best_neighbor_distance = current_distance
            
            for neighbor in neighbors:
                neighbor_distance = self.total_distance(neighbor)
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
            
            if best_neighbor is None:
                break
            
            current_route = best_neighbor
            current_distance = best_neighbor_distance
            iterations += 1
        
        return current_route, current_distance, iterations
    
    def random_restart(self, num_restarts):
        """Hill Climbing với Random Restart"""
        self.history = []
        self.best_distance = float('inf')
        self.best_route = None
        
        for restart in range(num_restarts):
            random_route = list(range(self.num_campuses))
            random.shuffle(random_route)
            
            route, distance, iterations = self.hill_climbing(random_route)
            
            self.history.append({
                'restart': restart + 1,
                'route': route,
                'distance': distance,
                'iterations': iterations
            })
            
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route
        
        return self.best_route, self.best_distance
    
    def get_statistics(self):
        """Tính các giá trị thống kê"""
        distances = [h['distance'] for h in self.history]
        return {
            'mean': statistics.mean(distances) if distances else 0,
            'std': statistics.stdev(distances) if len(distances) > 1 else 0,
            'best': self.best_distance
        }


class TSPGUI(QMainWindow):
    """Giao diện chính của chương trình"""
    
    def __init__(self):
        super().__init__()
        
        self.campuses = [
            ('CS1', (10.762622, 106.682308)),
            ('CS2', (10.729920, 106.693146)),
            ('A', (10.881173, 106.805991)),
            ('B', (10.850120, 106.771832)),
            ('C', (10.732432, 106.699287)),
            ('D', (10.870550, 106.803234)),
            ('E', (10.823099, 106.629537)),
            ('F', (10.838920, 106.756363))
        ]
        
        self.solver = TSPSolver(self.campuses)
        self.restart_index = 0
        self.animation_timer = None
        
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện người dùng"""
        self.setWindowTitle('TSP Solver - Hill Climbing với Random Restarts')
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel trái
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel('Số lần Random Restart:'))
        self.restart_spinbox = QSpinBox()
        self.restart_spinbox.setMinimum(1)
        self.restart_spinbox.setMaximum(100)
        self.restart_spinbox.setValue(10)
        control_layout.addWidget(self.restart_spinbox)
        
        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.run_solver)
        control_layout.addWidget(self.run_button)
        control_layout.addStretch()
        
        left_layout.addLayout(control_layout)
        
        left_layout.addWidget(QLabel('Solution Log:'))
        self.route_log = QTableWidget()
        self.route_log.setColumnCount(4)
        self.route_log.setHorizontalHeaderLabels(['Restart #', 'Route', 'Distance', 'Iterations'])
        self.route_log.setColumnWidth(1, 300)
        left_layout.addWidget(self.route_log)
        
        left_layout.addWidget(QLabel('Ma trận khoảng cách:'))
        self.distance_table = QTableWidget()
        self.distance_table.setRowCount(len(self.campuses))
        self.distance_table.setColumnCount(len(self.campuses))
        headers = [c[0] for c in self.campuses]
        self.distance_table.setHorizontalHeaderLabels(headers)
        self.distance_table.setVerticalHeaderLabels(headers)
        
        for i in range(len(self.campuses)):
            for j in range(len(self.campuses)):
                item = QTableWidgetItem(f'{self.solver.distance_matrix[i][j]:.4f}')
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.distance_table.setItem(i, j, item)
        
        left_layout.addWidget(self.distance_table)
        
        # Panel phải
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        stats_layout = QHBoxLayout()
        self.label_best = QLabel('Best Distance: -')
        self.label_mean = QLabel('Mean: -')
        self.label_std = QLabel('Std Dev: -')
        stats_layout.addWidget(self.label_best)
        stats_layout.addWidget(self.label_mean)
        stats_layout.addWidget(self.label_std)
        right_layout.addLayout(stats_layout)
        
        self.best_route_label = QLabel('Best Route: -')
        self.best_route_label.setWordWrap(True)
        right_layout.addWidget(self.best_route_label)
        
        self.fig_distribution = Figure(figsize=(5, 3))
        self.canvas_distribution = FigureCanvas(self.fig_distribution)
        right_layout.addWidget(self.canvas_distribution)
        
        self.fig_optimization = Figure(figsize=(5, 3))
        self.canvas_optimization = FigureCanvas(self.fig_optimization)
        right_layout.addWidget(self.canvas_optimization)
        
        self.fig_map = Figure(figsize=(5, 4))
        self.canvas_map = FigureCanvas(self.fig_map)
        right_layout.addWidget(self.canvas_map)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
    
    def run_solver(self):
        """Chạy thuật toán"""
        num_restarts = self.restart_spinbox.value()
        self.run_button.setEnabled(False)
        
        self.route_log.setRowCount(0)
        self.solver.random_restart(num_restarts)
        
        self.animate_route_plot()
    
    def animate_route_plot(self):
        """Khởi tạo animation cho route plot"""
        self.restart_index = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_route_animation)
        self.animation_timer.start(500)
    
    def update_route_animation(self):
        """Cập nhật animation từng bước"""
        if self.restart_index >= len(self.solver.history):
            self.animation_timer.stop()
            self.run_button.setEnabled(True)
            
            self.plot_route_with_color(self.solver.best_route, 'green')
            self.update_statistics_labels()
            
            route_names = [self.campuses[i][0] for i in self.solver.best_route]
            self.best_route_label.setText(f"Best Route: {' → '.join(route_names)}")
            return
        
        current = self.solver.history[self.restart_index]
        
        self.plot_specific_route(current['route'], self.restart_index)
        self.plot_distribution_step(self.restart_index)
        self.plot_optimization_step(self.restart_index)
        self.update_route_log(self.restart_index)
        
        self.restart_index += 1
    
    def plot_specific_route(self, route, index):
        """Vẽ một route cụ thể"""
        self.fig_map.clear()
        ax = self.fig_map.add_subplot(111)
        
        coords = [self.campuses[i][1] for i in route]
        coords.append(coords[0])
        
        x = [c[1] for c in coords]
        y = [c[0] for c in coords]
        
        ax.plot(x, y, 'bo-', markersize=8, linewidth=2)
        
        for i, idx in enumerate(route):
            name = self.campuses[idx][0]
            coord = self.campuses[idx][1]
            ax.text(coord[1], coord[0], f'  {name}', fontsize=10, ha='left')
        
        ax.set_title(f'Route at Restart #{index + 1}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        self.canvas_map.draw()
    
    def plot_route_with_color(self, route, color):
        """Vẽ route với màu tùy chỉnh"""
        self.fig_map.clear()
        ax = self.fig_map.add_subplot(111)
        
        coords = [self.campuses[i][1] for i in route]
        coords.append(coords[0])
        
        x = [c[1] for c in coords]
        y = [c[0] for c in coords]
        
        ax.plot(x, y, f'{color[0]}o-', markersize=8, linewidth=2, color=color)
        
        for i, idx in enumerate(route):
            name = self.campuses[idx][0]
            coord = self.campuses[idx][1]
            ax.text(coord[1], coord[0], f'  {name}', fontsize=10, ha='left', color='darkgreen')
        
        ax.set_title('Best Route Found')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        self.canvas_map.draw()
    
    def plot_distribution_step(self, index):
        """Vẽ histogram phân phối độ dài"""
        self.fig_distribution.clear()
        ax = self.fig_distribution.add_subplot(111)
        
        distances = [h['distance'] for h in self.solver.history[:index + 1]]
        
        ax.hist(distances, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title('Distance Distribution')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        self.canvas_distribution.draw()
    
    def plot_optimization_step(self, index):
        """Vẽ lịch sử tối ưu hóa"""
        self.fig_optimization.clear()
        ax = self.fig_optimization.add_subplot(111)
        
        distances = [h['distance'] for h in self.solver.history[:index + 1]]
        restarts = list(range(1, len(distances) + 1))
        
        ax.plot(restarts, distances, 'o-', color='coral', linewidth=2, markersize=6)
        ax.set_title('Optimization History')
        ax.set_xlabel('Restart #')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)
        
        self.canvas_optimization.draw()
    
    def update_route_log(self, index):
        """Cập nhật bảng log"""
        current = self.solver.history[index]
        
        row = self.route_log.rowCount()
        self.route_log.insertRow(row)
        
        route_names = [self.campuses[i][0] for i in current['route']]
        route_str = ' → '.join(route_names)
        
        self.route_log.setItem(row, 0, QTableWidgetItem(str(current['restart'])))
        self.route_log.setItem(row, 1, QTableWidgetItem(route_str))
        self.route_log.setItem(row, 2, QTableWidgetItem(f"{current['distance']:.4f}"))
        self.route_log.setItem(row, 3, QTableWidgetItem(str(current['iterations'])))
        
        self.route_log.scrollToBottom()
    
    def update_statistics_labels(self):
        """Cập nhật các label thống kê"""
        stats = self.solver.get_statistics()
        
        self.label_best.setText(f"Best Distance: {stats['best']:.4f}")
        self.label_mean.setText(f"Mean: {stats['mean']:.4f}")
        self.label_std.setText(f"Std Dev: {stats['std']:.4f}")


def main():
    app = QApplication(sys.argv)
    window = TSPGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()