# TSP Solver - Travelling Salesman Problem

Giải bài toán người du lịch (TSP)

## Mô tả

Chương trình giải bài toán TSP sử dụng 2 thuật toán:
- **Hill Climbing** với Random Restart
- **Ant Colony Optimization (ACO)**

## Cấu trúc thư mục

```
TSP_Project/
│
├── algorithms/           # Module chứa các thuật toán
│   ├── __init__.py
│   ├── hill_climbing.py # Thuật toán Hill Climbing
│   └── ant_colony.py    # Thuật toán ACO
│
├── gui/                 # Module giao diện
│   ├── __init__.py
│   └── main_window.py   # Giao diện chính PyQt5
│
├── utils/               # Module tiện ích
│   ├── __init__.py
│   ├── tsp_problem.py   # Class TSProblem
│   └── tsp_utils.py     # Các hàm tính toán
│
├── data/                # Dữ liệu mẫu
│   └── sample_cities.json
│
├── comparison.py        # Module so sánh thuật toán (NEW!)
├── demo_comparison.py   # Demo sử dụng comparison.py (NEW!)
├── config.py            # Cấu hình
├── main.py              # Entry point cho GUI
├── requirements.txt     # Dependencies
├── BAO_CAO_DO_AN.md     # Báo cáo đồ án đầy đủ
├── BAO_CAO_THUAT_TOAN.md # Báo cáo phần thuật toán chi tiết
├── COMPARISON_README.md # Hướng dẫn module comparison
└── README.md            # File này
```

## Cài đặt

### 1. Cài đặt Python
Yêu cầu Python 3.7 trở lên

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Chạy chương trình

### GUI (Giao diện đồ họa)
```bash
python main.py
```

### Command Line (So sánh thuật toán)
```bash
# So sánh cơ bản với 20 thành phố
python comparison.py -n 20

# So sánh với tham số tùy chỉnh
python comparison.py -n 30 --hc_iterations 5000 --aco_ants 30 --save

# Chạy demo tương tác
python demo_comparison.py
```

## Hướng dẫn sử dụng

### 1. Tạo dữ liệu
- **Sinh ngẫu nhiên**: Nhập số thành phố và nhấn "Sinh ngẫu nhiên"
- **Tải từ file**: Nhấn "Tải file" và chọn file JSON
- **Lưu vào file**: Nhấn "Lưu file" để lưu dữ liệu hiện tại

### 2. Cấu hình thuật toán

#### Hill Climbing
- **Số lần lặp**: Số iteration tối đa (mặc định: 10000)
- **Restart limit**: Số lần không cải thiện trước khi restart (mặc định: 50)

#### Ant Colony Optimization
- **Số kiến**: Số lượng kiến trong mỗi thế hệ (mặc định: 50)
- **Số thế hệ**: Số iteration (mặc định: 100)
- **Alpha**: Tầm quan trọng của pheromone (mặc định: 1.0)
- **Beta**: Tầm quan trọng của heuristic (mặc định: 5.0)
- **Evaporation**: Tỷ lệ bay hơi pheromone (mặc định: 0.5)

### 3. Chạy thuật toán
- Nhấn **"Chạy Hill Climbing"** hoặc **"Chạy ACO"** để chạy từng thuật toán
- Nhấn **"So sánh cả hai thuật toán"** để chạy và so sánh cả hai

### 4. Xem kết quả
- **Tab Hành trình**: Hiển thị đường đi tối ưu
- **Tab Đồ thị hội tụ**: Hiển thị quá trình tối ưu hóa
- **Tab So sánh**: So sánh hiệu năng của hai thuật toán

## Chi tiết thuật toán

### Hill Climbing
- Thuật toán tìm kiếm cục bộ
- Sử dụng 2-opt để tạo các nghiệm lân cận
- Chọn nghiệm lân cận tốt nhất (steepest ascent)
- Random restart khi gặp local optimum

### Ant Colony Optimization
- Thuật toán mô phỏng hành vi đàn kiến
- Kiến xây dựng nghiệm dựa trên pheromone và heuristic
- Pheromone cập nhật theo chất lượng nghiệm
- Pheromone bay hơi theo thời gian

## Format file JSON

```json
{
  "0": [x0, y0],
  "1": [x1, y1],
  "2": [x2, y2],
  ...
}
```

Ví dụ: `data/sample_cities.json`

## Tính năng

### GUI
- ✅ Sinh ngẫu nhiên thành phố  
- ✅ Tải/Lưu dữ liệu từ file JSON  
- ✅ Thuật toán Hill Climbing với Random Restart  
- ✅ Thuật toán Ant Colony Optimization
- ✅ Visualization hành trình và đồ thị hội tụ
- ✅ So sánh trực quan giữa hai thuật toán

### Command Line (comparison.py)
- ✅ So sánh tự động giữa Hill Climbing và ACO
- ✅ Phân tích chi tiết kết quả
- ✅ Vẽ biểu đồ so sánh (distance, time, fitness, convergence)
- ✅ Vẽ tours của cả hai thuật toán
- ✅ Lưu kết quả ra JSON
- ✅ Tùy chỉnh đầy đủ tham số
- ✅ Chạy độc lập không cần GUI

## Module mới: comparison.py

Module `comparison.py` cho phép so sánh hiệu năng giữa Hill Climbing và ACO một cách tự động và chi tiết.

### Sử dụng trong code:
```python
from comparison import AlgorithmComparison
from utils.tsp_problem import TSProblem

# Tạo problem
tsp = TSProblem(num_cities=20)

# So sánh
comparison = AlgorithmComparison(tsp)
result = comparison.compare(verbose=True)

# Vẽ biểu đồ
comparison.plot_comparison(save_path='comparison.png')
comparison.plot_tours(save_path='tours.png')

# Lưu kết quả
comparison.save_results('results.json')
```

-Hiển thị trực quan hành trình  
-Đồ thị hội tụ  
-So sánh hiệu năng hai thuật toán  
-Giao diện thân thiện với PyQt5  

