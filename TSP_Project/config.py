"""
Cấu hình cho ứng dụng TSP
"""

# Cấu hình Hill Climbing
HILL_CLIMBING_CONFIG = {
    'max_iterations': 10000,
    'max_restarts': 10  # Số lần khởi động lại (random restart)
}

# Cấu hình Ant Colony Optimization
ACO_CONFIG = {
    'num_ants': 50,
    'num_iterations': 100,
    'alpha': 1.0,      # Tầm quan trọng của pheromone
    'beta': 5.0,       # Tầm quan trọng của thông tin heuristic
    'evaporation': 0.5,  # Tỷ lệ bay hơi pheromone
    'q': 100           # Hằng số pheromone
}

# Cấu hình GUI
GUI_CONFIG = {
    'window_width': 1400,
    'window_height': 900,
    'plot_width': 800,
    'plot_height': 600
}
