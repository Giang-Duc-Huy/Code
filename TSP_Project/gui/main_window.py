
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSpinBox, QGroupBox, 
                             QTextEdit, QFileDialog, QTabWidget, QDoubleSpinBox,
                             QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import json

from algorithms.hill_climbing import HillClimbing
from algorithms.ant_colony import AntColonyOptimization
from utils.tsp_problem import TSProblem
from config import HILL_CLIMBING_CONFIG, ACO_CONFIG
import numpy as np


class AlgorithmThread(QThread):
    """Thread để chạy thuật toán không block GUI"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    
    def __init__(self, algorithm, algorithm_type):
        super().__init__()
        self.algorithm = algorithm
        self.algorithm_type = algorithm_type
    
    def run(self):
        """Chạy thuật toán"""
        try:
            self.progress.emit(f"Đang chạy {self.algorithm_type}...")
            result = self.algorithm.solve(verbose=False)
            result['algorithm'] = self.algorithm_type
            self.finished.emit(result)
        except Exception as e:
            self.progress.emit(f"Lỗi: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tsp_problem = None
        self.results = {}
        self.current_thread = None
        
        self.initUI()
    
    def initUI(self):
        """Khởi tạo giao diện"""
        self.setWindowTitle('TSP Solver - Hill Climbing & Ant Colony Optimization')
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Panel bên trái - Điều khiển
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel bên phải - Visualization
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_control_panel(self):
        """Tạo panel điều khiển"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tiêu đề
        title = QLabel('GIẢI BÀI TOÁN NGƯỜI DU LỊCH')
        title.setFont(QFont('Arial', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Group: Dữ liệu
        data_group = QGroupBox('Dữ liệu thành phố')
        data_layout = QVBoxLayout()
        
        # Sinh ngẫu nhiên
        random_layout = QHBoxLayout()
        random_layout.addWidget(QLabel('Số thành phố:'))
        self.num_cities_spin = QSpinBox()
        self.num_cities_spin.setRange(4, 100)
        self.num_cities_spin.setValue(20)
        random_layout.addWidget(self.num_cities_spin)
        data_layout.addLayout(random_layout)
        
        self.btn_generate = QPushButton('Sinh ngẫu nhiên')
        self.btn_generate.clicked.connect(self.generate_cities)
        data_layout.addWidget(self.btn_generate)
        
        # Load/Save file
        file_layout = QHBoxLayout()
        self.btn_load = QPushButton('Tải file')
        self.btn_load.clicked.connect(self.load_cities)
        file_layout.addWidget(self.btn_load)
        
        self.btn_save = QPushButton('Lưu file')
        self.btn_save.clicked.connect(self.save_cities)
        file_layout.addWidget(self.btn_save)
        data_layout.addLayout(file_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Group: Hill Climbing
        hc_group = QGroupBox('Hill Climbing')
        hc_layout = QVBoxLayout()
        
        hc_iter_layout = QHBoxLayout()
        hc_iter_layout.addWidget(QLabel('Số lần lặp:'))
        self.hc_iterations = QSpinBox()
        self.hc_iterations.setRange(100, 50000)
        self.hc_iterations.setValue(HILL_CLIMBING_CONFIG['max_iterations'])
        hc_iter_layout.addWidget(self.hc_iterations)
        hc_layout.addLayout(hc_iter_layout)
        
        hc_restart_layout = QHBoxLayout()
        hc_restart_layout.addWidget(QLabel('Số lần restart:'))
        self.hc_restart = QSpinBox()
        self.hc_restart.setRange(1, 50)
        self.hc_restart.setValue(HILL_CLIMBING_CONFIG['max_restarts'])
        hc_restart_layout.addWidget(self.hc_restart)
        hc_layout.addLayout(hc_restart_layout)
        
        self.btn_run_hc = QPushButton('Chạy Hill Climbing')
        self.btn_run_hc.clicked.connect(self.run_hill_climbing)
        hc_layout.addWidget(self.btn_run_hc)
        
        hc_group.setLayout(hc_layout)
        layout.addWidget(hc_group)
        
        # Group: ACO
        aco_group = QGroupBox('Ant Colony Optimization')
        aco_layout = QVBoxLayout()
        
        aco_ants_layout = QHBoxLayout()
        aco_ants_layout.addWidget(QLabel('Số kiến:'))
        self.aco_ants = QSpinBox()
        self.aco_ants.setRange(10, 200)
        self.aco_ants.setValue(ACO_CONFIG['num_ants'])
        aco_ants_layout.addWidget(self.aco_ants)
        aco_layout.addLayout(aco_ants_layout)
        
        aco_iter_layout = QHBoxLayout()
        aco_iter_layout.addWidget(QLabel('Số thế hệ:'))
        self.aco_iterations = QSpinBox()
        self.aco_iterations.setRange(10, 500)
        self.aco_iterations.setValue(ACO_CONFIG['num_iterations'])
        aco_iter_layout.addWidget(self.aco_iterations)
        aco_layout.addLayout(aco_iter_layout)
        
        aco_alpha_layout = QHBoxLayout()
        aco_alpha_layout.addWidget(QLabel('Alpha:'))
        self.aco_alpha = QDoubleSpinBox()
        self.aco_alpha.setRange(0.1, 5.0)
        self.aco_alpha.setSingleStep(0.1)
        self.aco_alpha.setValue(ACO_CONFIG['alpha'])
        aco_alpha_layout.addWidget(self.aco_alpha)
        aco_layout.addLayout(aco_alpha_layout)
        
        aco_beta_layout = QHBoxLayout()
        aco_beta_layout.addWidget(QLabel('Beta:'))
        self.aco_beta = QDoubleSpinBox()
        self.aco_beta.setRange(0.1, 10.0)
        self.aco_beta.setSingleStep(0.1)
        self.aco_beta.setValue(ACO_CONFIG['beta'])
        aco_beta_layout.addWidget(self.aco_beta)
        aco_layout.addLayout(aco_beta_layout)
        
        aco_evap_layout = QHBoxLayout()
        aco_evap_layout.addWidget(QLabel('Evaporation:'))
        self.aco_evaporation = QDoubleSpinBox()
        self.aco_evaporation.setRange(0.1, 0.9)
        self.aco_evaporation.setSingleStep(0.05)
        self.aco_evaporation.setValue(ACO_CONFIG['evaporation'])
        aco_evap_layout.addWidget(self.aco_evaporation)
        aco_layout.addLayout(aco_evap_layout)
        
        self.btn_run_aco = QPushButton('Chạy ACO')
        self.btn_run_aco.clicked.connect(self.run_aco)
        aco_layout.addWidget(self.btn_run_aco)
        
        aco_group.setLayout(aco_layout)
        layout.addWidget(aco_group)
        
        # Button so sánh
        self.btn_compare = QPushButton('So sánh cả hai thuật toán')
        self.btn_compare.clicked.connect(self.compare_algorithms)
        self.btn_compare.setStyleSheet('background-color: #4CAF50; color: white; font-weight: bold;')
        layout.addWidget(self.btn_compare)
        
        # Kết quả
        result_group = QGroupBox('Kết quả')
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        result_layout.addWidget(self.result_text)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Tạo panel hiển thị đồ họa"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Tabs để hiển thị nhiều đồ thị
        self.tabs = QTabWidget()
        
        # Tab 1: Tour visualization
        self.figure_tour = Figure(figsize=(8, 6))
        self.canvas_tour = FigureCanvas(self.figure_tour)
        self.tabs.addTab(self.canvas_tour, 'Hành trình')
        
        # Tab 2: Convergence graph
        self.figure_convergence = Figure(figsize=(8, 6))
        self.canvas_convergence = FigureCanvas(self.figure_convergence)
        self.tabs.addTab(self.canvas_convergence, 'Đồ thị hội tụ')
        
        # Tab 3: Comparison
        self.figure_comparison = Figure(figsize=(8, 6))
        self.canvas_comparison = FigureCanvas(self.figure_comparison)
        self.tabs.addTab(self.canvas_comparison, 'So sánh')
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def generate_cities(self):
        """Sinh ngẫu nhiên các thành phố"""
        num_cities = self.num_cities_spin.value()
        try:
            self.tsp_problem = TSProblem(num_cities=num_cities)
            self.result_text.append(f'Đã sinh {num_cities} thành phố ngẫu nhiên.')
            self.plot_cities()
        except Exception as e:
            QMessageBox.warning(self, 'Lỗi', str(e))
    
    def load_cities(self):
        """Tải dữ liệu thành phố từ file"""
        filename, _ = QFileDialog.getOpenFileName(self, 'Tải file', '', 'JSON Files (*.json)')
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Chuyển đổi từ dict sang array tọa độ
                city_coords = []
                for key in sorted(data.keys(), key=lambda x: int(x)):
                    city_coords.append(data[key])
                
                self.tsp_problem = TSProblem(city_coords=city_coords)
                self.result_text.append(f'Đã tải {len(city_coords)} thành phố từ {filename}')
                self.plot_cities()
            except Exception as e:
                QMessageBox.warning(self, 'Lỗi', f'Không thể tải file: {str(e)}')
    
    def save_cities(self):
        """Lưu dữ liệu thành phố vào file"""
        if self.tsp_problem is None:
            QMessageBox.warning(self, 'Lỗi', 'Chưa có dữ liệu thành phố!')
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, 'Lưu file', '', 'JSON Files (*.json)')
        if filename:
            try:
                coords = self.tsp_problem.get_city_coords()
                data = {str(i): coords[i].tolist() for i in range(len(coords))}
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                self.result_text.append(f'Đã lưu dữ liệu vào {filename}')
            except Exception as e:
                QMessageBox.warning(self, 'Lỗi', f'Không thể lưu file: {str(e)}')
    
    def plot_cities(self):
        """Vẽ các thành phố"""
        if self.tsp_problem is None:
            return
        
        self.figure_tour.clear()
        ax = self.figure_tour.add_subplot(111)
        
        coords = self.tsp_problem.get_city_coords()
        x = coords[:, 0]
        y = coords[:, 1]
        
        ax.scatter(x, y, c='red', s=100, zorder=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Vị trí {self.tsp_problem.num_cities} thành phố')
        ax.grid(True, alpha=0.3)
        
        self.canvas_tour.draw()
    
    def plot_tour(self, tour, title, color='blue'):
        """Vẽ hành trình"""
        if not tour or self.tsp_problem is None:
            return
        
        self.figure_tour.clear()
        ax = self.figure_tour.add_subplot(111)
        
        coords = self.tsp_problem.get_city_coords()
        
        # Vẽ các thành phố
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=2)
        
        # Vẽ hành trình
        for i in range(len(tour)):
            city1 = coords[tour[i]]
            city2 = coords[tour[(i + 1) % len(tour)]]
            ax.plot([city1[0], city2[0]], [city1[1], city2[1]], 
                   c=color, linewidth=2, alpha=0.6, zorder=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        self.canvas_tour.draw()
    
    def plot_convergence(self, history, algorithm_name):
        """Vẽ đồ thị hội tụ"""
        if not history:
            return
        
        self.figure_convergence.clear()
        ax = self.figure_convergence.add_subplot(111)
        
        iterations = [h['iteration'] for h in history]
        
        if algorithm_name == 'Hill Climbing':
            distances = [h['best_distance'] for h in history]
            ax.plot(iterations, distances, 'b-', linewidth=2, label='Best Distance')
        else:  # ACO
            best_distances = [h['best_distance'] for h in history]
            avg_distances = [h['avg_distance'] for h in history]
            ax.plot(iterations, best_distances, 'b-', linewidth=2, label='Best Distance')
            ax.plot(iterations, avg_distances, 'r--', linewidth=1, label='Average Distance', alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance')
        ax.set_title(f'Quá trình hội tụ - {algorithm_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas_convergence.draw()
    
    def run_hill_climbing(self):
        """Chạy thuật toán Hill Climbing"""
        if self.tsp_problem is None:
            QMessageBox.warning(self, 'Lỗi', 'Chưa có dữ liệu thành phố!')
            return
        
        self.disable_buttons()
        self.result_text.append('\n--- Bắt đầu Hill Climbing ---')
        
        hc = HillClimbing(
            self.tsp_problem,
            max_iterations=self.hc_iterations.value(),
            max_restarts=self.hc_restart.value()
        )
        
        self.current_thread = AlgorithmThread(hc, 'Hill Climbing')
        self.current_thread.finished.connect(self.on_algorithm_finished)
        self.current_thread.progress.connect(self.result_text.append)
        self.current_thread.start()
    
    def run_aco(self):
        """Chạy thuật toán ACO"""
        if self.tsp_problem is None:
            QMessageBox.warning(self, 'Lỗi', 'Chưa có dữ liệu thành phố!')
            return
        
        self.disable_buttons()
        self.result_text.append('\n--- Bắt đầu Ant Colony Optimization ---')
        
        aco = AntColonyOptimization(
            self.tsp_problem,
            num_ants=self.aco_ants.value(),
            num_iterations=self.aco_iterations.value(),
            alpha=self.aco_alpha.value(),
            beta=self.aco_beta.value(),
            evaporation=self.aco_evaporation.value()
        )
        
        self.current_thread = AlgorithmThread(aco, 'ACO')
        self.current_thread.finished.connect(self.on_algorithm_finished)
        self.current_thread.progress.connect(self.result_text.append)
        self.current_thread.start()
    
    def on_algorithm_finished(self, result):
        """Xử lý khi thuật toán chạy xong"""
        algorithm = result['algorithm']
        self.results[algorithm] = result
        
        # Hiển thị kết quả
        self.result_text.append(f'\nKết quả {algorithm}:')
        self.result_text.append(f'  Khoảng cách: {result["distance"]:.2f}')
        self.result_text.append(f'  Thời gian: {result["time"]:.3f} giây')
        
        # Vẽ hành trình
        color = 'blue' if algorithm == 'Hill Climbing' else 'green'
        self.plot_tour(result['tour'], 
                      f'{algorithm} - Distance: {result["distance"]:.2f}', 
                      color)
        
        # Vẽ đồ thị hội tụ
        self.plot_convergence(result['history'], algorithm)
        
        self.enable_buttons()
    
    def compare_algorithms(self):
        """So sánh cả hai thuật toán"""
        if self.tsp_problem is None:
            QMessageBox.warning(self, 'Lỗi', 'Chưa có dữ liệu thành phố!')
            return
        
        self.result_text.append('\n=== SO SÁNH CẢ HAI THUẬT TOÁN ===')
        
        # Chạy Hill Climbing
        self.result_text.append('Đang chạy Hill Climbing...')
        hc = HillClimbing(
            self.tsp_problem,
            max_iterations=self.hc_iterations.value(),
            max_restarts=self.hc_restart.value()
        )
        hc_result = hc.solve(verbose=False)
        self.results['Hill Climbing'] = hc_result
        
        # Chạy ACO
        self.result_text.append('Đang chạy ACO...')
        aco = AntColonyOptimization(
            self.tsp_problem,
            num_ants=self.aco_ants.value(),
            num_iterations=self.aco_iterations.value(),
            alpha=self.aco_alpha.value(),
            beta=self.aco_beta.value(),
            evaporation=self.aco_evaporation.value()
        )
        aco_result = aco.solve(verbose=False)
        self.results['ACO'] = aco_result
        
        # Hiển thị kết quả so sánh
        self.result_text.append('\n--- KẾT QUẢ SO SÁNH ---')
        self.result_text.append(f'Hill Climbing:')
        self.result_text.append(f'  Khoảng cách: {hc_result["distance"]:.2f}')
        self.result_text.append(f'  Thời gian: {hc_result["time"]:.3f} giây')
        self.result_text.append(f'\nACO:')
        self.result_text.append(f'  Khoảng cách: {aco_result["distance"]:.2f}')
        self.result_text.append(f'  Thời gian: {aco_result["time"]:.3f} giây')
        
        # Xác định thuật toán tốt hơn
        if hc_result["distance"] < aco_result["distance"]:
            self.result_text.append(f'\n✓ Hill Climbing tốt hơn (ngắn hơn {aco_result["distance"] - hc_result["distance"]:.2f})')
        elif aco_result["distance"] < hc_result["distance"]:
            self.result_text.append(f'\n✓ ACO tốt hơn (ngắn hơn {hc_result["distance"] - aco_result["distance"]:.2f})')
        else:
            self.result_text.append('\n✓ Cả hai cho kết quả tương đương')
        
        # Vẽ biểu đồ so sánh
        self.plot_comparison()
    
    def plot_comparison(self):
        """Vẽ biểu đồ so sánh"""
        if 'Hill Climbing' not in self.results or 'ACO' not in self.results:
            return
        
        self.figure_comparison.clear()
        
        # Subplot 1: Distance comparison
        ax1 = self.figure_comparison.add_subplot(121)
        algorithms = ['Hill Climbing', 'ACO']
        distances = [self.results['Hill Climbing']['distance'], 
                    self.results['ACO']['distance']]
        colors = ['blue', 'green']
        
        ax1.bar(algorithms, distances, color=colors, alpha=0.7)
        ax1.set_ylabel('Khoảng cách')
        ax1.set_title('So sánh khoảng cách')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Time comparison
        ax2 = self.figure_comparison.add_subplot(122)
        times = [self.results['Hill Climbing']['time'], 
                self.results['ACO']['time']]
        
        ax2.bar(algorithms, times, color=colors, alpha=0.7)
        ax2.set_ylabel('Thời gian (giây)')
        ax2.set_title('So sánh thời gian')
        ax2.grid(True, alpha=0.3, axis='y')
        
        self.figure_comparison.tight_layout()
        self.canvas_comparison.draw()
        
        # Chuyển sang tab so sánh
        self.tabs.setCurrentIndex(2)
    
    def disable_buttons(self):
        """Vô hiệu hóa các nút khi đang chạy"""
        self.btn_run_hc.setEnabled(False)
        self.btn_run_aco.setEnabled(False)
        self.btn_compare.setEnabled(False)
    
    def enable_buttons(self):
        """Kích hoạt lại các nút"""
        self.btn_run_hc.setEnabled(True)
        self.btn_run_aco.setEnabled(True)
        self.btn_compare.setEnabled(True)
