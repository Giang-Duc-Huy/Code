"""
Module so sánh hiệu năng giữa Hill Climbing và ACO
"""

import time
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

from utils.tsp_problem import TSProblem
from algorithms.hill_climbing import HillClimbing
from algorithms.ant_colony import AntColonyOptimization


class AlgorithmComparison:
    
    def __init__(self, tsp_problem: TSProblem):
      
        self.tsp_problem = tsp_problem
        self.results = {}
    
    def run_hill_climbing(self, max_iterations=10000, max_restarts=10, verbose=True):
      
        if verbose:
            print("\n" + "="*50)
            print("HILL CLIMBING")
            print("="*50)
        
        hc = HillClimbing(
            self.tsp_problem,
            max_iterations=max_iterations,
            max_restarts=max_restarts
        )
        
        start_time = time.time()
        result = hc.solve(verbose=verbose)
        end_time = time.time()
        
        result['time'] = end_time - start_time
        result['algorithm'] = 'Hill Climbing'
        
        self.results['Hill Climbing'] = result
        
        if verbose:
            print(f"\n✓ Hoàn thành trong {result['time']:.3f} giây")
        
        return result
    
    def run_aco(self, num_ants=50, num_iterations=100, 
                alpha=1.0, beta=5.0, evaporation=0.5, verbose=True):
       
        if verbose:
            print("\n" + "="*50)
            print("ANT COLONY OPTIMIZATION")
            print("="*50)
        
        aco = AntColonyOptimization(
            self.tsp_problem,
            num_ants=num_ants,
            num_iterations=num_iterations,
            alpha=alpha,
            beta=beta,
            evaporation=evaporation
        )
        
        start_time = time.time()
        result = aco.solve(verbose=verbose)
        end_time = time.time()
        
        result['time'] = end_time - start_time
        result['algorithm'] = 'ACO'
        
        self.results['ACO'] = result
        
        if verbose:
            print(f"\n✓ Hoàn thành trong {result['time']:.3f} giây")
        
        return result
    
    def compare(self, hc_config=None, aco_config=None, verbose=True):
       
        # Default configurations
        if hc_config is None:
            hc_config = {'max_iterations': 10000, 'max_restarts': 10}
        if aco_config is None:
            aco_config = {
                'num_ants': 50,
                'num_iterations': 100,
                'alpha': 1.0,
                'beta': 5.0,
                'evaporation': 0.5
            }
        
        # Chạy Hill Climbing
        hc_result = self.run_hill_climbing(**hc_config, verbose=verbose)
        
        # Chạy ACO
        aco_result = self.run_aco(**aco_config, verbose=verbose)
        
        # Phân tích kết quả
        comparison = self.analyze_results(hc_result, aco_result)
        
        if verbose:
            self.print_comparison(comparison)
        
        return comparison
    
    def analyze_results(self, hc_result: Dict, aco_result: Dict) -> Dict[str, Any]:
      
        comparison = {
            'hill_climbing': {
                'distance': hc_result['distance'],
                'fitness': hc_result.get('fitness', 0),
                'time': hc_result['time'],
                'tour': hc_result['tour']
            },
            'aco': {
                'distance': aco_result['distance'],
                'fitness': aco_result.get('fitness', 0),
                'time': aco_result['time'],
                'tour': aco_result['tour']
            },
            'analysis': {}
        }
        
        # So sánh distance
        distance_diff = hc_result['distance'] - aco_result['distance']
        if abs(distance_diff) < 0.01:
            comparison['analysis']['distance_winner'] = 'Tie'
            comparison['analysis']['distance_message'] = 'Kết quả tương đương'
        elif distance_diff > 0:
            comparison['analysis']['distance_winner'] = 'ACO'
            improvement = (distance_diff / hc_result['distance']) * 100
            comparison['analysis']['distance_message'] = f'ACO tốt hơn {improvement:.2f}% (ngắn hơn {distance_diff:.2f})'
        else:
            comparison['analysis']['distance_winner'] = 'Hill Climbing'
            improvement = (abs(distance_diff) / aco_result['distance']) * 100
            comparison['analysis']['distance_message'] = f'Hill Climbing tốt hơn {improvement:.2f}% (ngắn hơn {abs(distance_diff):.2f})'
        
        # So sánh time
        time_diff = hc_result['time'] - aco_result['time']
        if abs(time_diff) < 0.01:
            comparison['analysis']['time_winner'] = 'Tie'
            comparison['analysis']['time_message'] = 'Thời gian tương đương'
        elif time_diff > 0:
            comparison['analysis']['time_winner'] = 'ACO'
            comparison['analysis']['time_message'] = f'ACO nhanh hơn {abs(time_diff):.3f}s'
        else:
            comparison['analysis']['time_winner'] = 'Hill Climbing'
            comparison['analysis']['time_message'] = f'Hill Climbing nhanh hơn {abs(time_diff):.3f}s'
        
        # Tổng kết
        if comparison['analysis']['distance_winner'] == comparison['analysis']['time_winner']:
            comparison['analysis']['overall_winner'] = comparison['analysis']['distance_winner']
        else:
            # Ưu tiên chất lượng nghiệm hơn tốc độ
            comparison['analysis']['overall_winner'] = comparison['analysis']['distance_winner']
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any]):
      
        print("\n" + "="*70)
        print("KẾT QUẢ SO SÁNH")
        print("="*70)
        
        # Hill Climbing
        print("\n📊 HILL CLIMBING:")
        print(f"  ├─ Khoảng cách: {comparison['hill_climbing']['distance']:.2f}")
        print(f"  ├─ Fitness:     {comparison['hill_climbing']['fitness']:.6f}")
        print(f"  └─ Thời gian:   {comparison['hill_climbing']['time']:.3f}s")
        
        # ACO
        print("\n📊 ANT COLONY OPTIMIZATION:")
        print(f"  ├─ Khoảng cách: {comparison['aco']['distance']:.2f}")
        print(f"  ├─ Fitness:     {comparison['aco']['fitness']:.6f}")
        print(f"  └─ Thời gian:   {comparison['aco']['time']:.3f}s")
        
        # Analysis
        print("\n" + "-"*70)
        print("PHÂN TÍCH:")
        print(f"  🏆 Chất lượng nghiệm: {comparison['analysis']['distance_message']}")
        print(f"  ⚡ Tốc độ thực thi:   {comparison['analysis']['time_message']}")
        print(f"  ✨ Tổng kết: {comparison['analysis']['overall_winner']} thắng!")
        print("="*70)
    
    def plot_comparison(self, save_path=None):
       
        if 'Hill Climbing' not in self.results or 'ACO' not in self.results:
            print("⚠️  Chưa có kết quả để so sánh!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('So sánh Hill Climbing vs ACO', fontsize=16, fontweight='bold')
        
        algorithms = ['Hill Climbing', 'ACO']
        colors = ['#3498db', '#2ecc71']
        
        # 1. Distance comparison
        ax1 = axes[0, 0]
        distances = [self.results['Hill Climbing']['distance'], 
                    self.results['ACO']['distance']]
        bars1 = ax1.bar(algorithms, distances, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Khoảng cách', fontsize=12)
        ax1.set_title('So sánh Khoảng cách Tour', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, distance in zip(bars1, distances):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{distance:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Time comparison
        ax2 = axes[0, 1]
        times = [self.results['Hill Climbing']['time'], 
                self.results['ACO']['time']]
        bars2 = ax2.bar(algorithms, times, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Thời gian (giây)', fontsize=12)
        ax2.set_title('So sánh Thời gian Thực thi', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Fitness comparison
        ax3 = axes[1, 0]
        fitness_values = [self.results['Hill Climbing'].get('fitness', 0), 
                         self.results['ACO'].get('fitness', 0)]
        bars3 = ax3.bar(algorithms, fitness_values, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Fitness (1/distance)', fontsize=12)
        ax3.set_title('So sánh Fitness', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, fitness in zip(bars3, fitness_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{fitness:.6f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 4. Convergence comparison
        ax4 = axes[1, 1]
        if 'history' in self.results['Hill Climbing'] and self.results['Hill Climbing']['history']:
            hc_history = self.results['Hill Climbing']['history']
            hc_iterations = [h['iteration'] for h in hc_history]
            hc_distances = [h['distance'] for h in hc_history]
            ax4.plot(hc_iterations, hc_distances, label='Hill Climbing', 
                    color=colors[0], linewidth=2, marker='o', markersize=3, alpha=0.7)
        
        if 'history' in self.results['ACO'] and self.results['ACO']['history']:
            aco_history = self.results['ACO']['history']
            aco_iterations = [h['iteration'] for h in aco_history]
            aco_distances = [h['best_distance'] for h in aco_history]
            ax4.plot(aco_iterations, aco_distances, label='ACO', 
                    color=colors[1], linewidth=2, marker='s', markersize=3, alpha=0.7)
        
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Khoảng cách', fontsize=12)
        ax4.set_title('Đồ thị Hội tụ', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Đã lưu biểu đồ vào {save_path}")
        
        plt.show()
    
    def plot_tours(self, save_path=None):
       
        if 'Hill Climbing' not in self.results or 'ACO' not in self.results:
            print("⚠️  Chưa có kết quả để vẽ!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('So sánh Tours', fontsize=16, fontweight='bold')
        
        coords = self.tsp_problem.get_city_coords()
        
        # Hill Climbing tour
        ax1 = axes[0]
        hc_tour = self.results['Hill Climbing']['tour']
        hc_distance = self.results['Hill Climbing']['distance']
        
        # Plot cities
        ax1.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3, 
                   edgecolors='black', linewidths=1.5, label='Thành phố')
        
        # Plot tour
        for i in range(len(hc_tour)):
            city1 = coords[hc_tour[i]]
            city2 = coords[hc_tour[(i + 1) % len(hc_tour)]]
            ax1.plot([city1[0], city2[0]], [city1[1], city2[1]], 
                    'b-', linewidth=2, alpha=0.6, zorder=1)
            ax1.arrow(city1[0], city1[1], 
                     (city2[0] - city1[0]) * 0.3, 
                     (city2[1] - city1[1]) * 0.3,
                     head_width=2, head_length=2, fc='blue', ec='blue', 
                     alpha=0.4, zorder=2)
        
        ax1.set_title(f'Hill Climbing\nDistance: {hc_distance:.2f}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()
        
        # ACO tour
        ax2 = axes[1]
        aco_tour = self.results['ACO']['tour']
        aco_distance = self.results['ACO']['distance']
        
        # Plot cities
        ax2.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3,
                   edgecolors='black', linewidths=1.5, label='Thành phố')
        
        # Plot tour
        for i in range(len(aco_tour)):
            city1 = coords[aco_tour[i]]
            city2 = coords[aco_tour[(i + 1) % len(aco_tour)]]
            ax2.plot([city1[0], city2[0]], [city1[1], city2[1]], 
                    'g-', linewidth=2, alpha=0.6, zorder=1)
            ax2.arrow(city1[0], city1[1], 
                     (city2[0] - city1[0]) * 0.3, 
                     (city2[1] - city1[1]) * 0.3,
                     head_width=2, head_length=2, fc='green', ec='green', 
                     alpha=0.4, zorder=2)
        
        ax2.set_title(f'Ant Colony Optimization\nDistance: {aco_distance:.2f}', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Đã lưu tours vào {save_path}")
        
        plt.show()
    
    def save_results(self, filename='comparison_results.json'):
    
        if not self.results:
            print("⚠️  Chưa có kết quả để lưu!")
            return
        
        # Prepare data for JSON (convert numpy types)
        data = {}
        for algo_name, result in self.results.items():
            data[algo_name] = {
                'distance': float(result['distance']),
                'fitness': float(result.get('fitness', 0)),
                'time': float(result['time']),
                'tour': [int(x) for x in result['tour']],
                'num_cities': len(result['tour'])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Đã lưu kết quả vào {filename}")


def main():
   
    import argparse
    
    parser = argparse.ArgumentParser(description='So sánh Hill Climbing và ACO cho TSP')
    parser.add_argument('-n', '--num_cities', type=int, default=20,
                       help='Số lượng thành phố (default: 20)')
    parser.add_argument('--hc_iterations', type=int, default=10000,
                       help='Hill Climbing: số iterations (default: 10000)')
    parser.add_argument('--hc_restarts', type=int, default=10,
                       help='Hill Climbing: số restarts (default: 10)')
    parser.add_argument('--aco_ants', type=int, default=50,
                       help='ACO: số kiến (default: 50)')
    parser.add_argument('--aco_iterations', type=int, default=100,
                       help='ACO: số iterations (default: 100)')
    parser.add_argument('--save', action='store_true',
                       help='Lưu kết quả và biểu đồ')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SO SÁNH THUẬT TOÁN GIẢI TSP")
    print("="*70)
    print(f"Số thành phố: {args.num_cities}")
    print(f"Hill Climbing config: iterations={args.hc_iterations}, restarts={args.hc_restarts}")
    print(f"ACO config: ants={args.aco_ants}, iterations={args.aco_iterations}")
    
    # Tạo TSP problem
    tsp = TSProblem(num_cities=args.num_cities)
    
    # Tạo comparison
    comparison = AlgorithmComparison(tsp)
    
    # Cấu hình
    hc_config = {
        'max_iterations': args.hc_iterations,
        'max_restarts': args.hc_restarts
    }
    
    aco_config = {
        'num_ants': args.aco_ants,
        'num_iterations': args.aco_iterations
    }
    
    # So sánh
    result = comparison.compare(hc_config=hc_config, aco_config=aco_config)
    
    # Vẽ biểu đồ
    comparison.plot_comparison(save_path='comparison_chart.png' if args.save else None)
    comparison.plot_tours(save_path='tours_comparison.png' if args.save else None)
    
    # Lưu kết quả
    if args.save:
        comparison.save_results('comparison_results.json')


if __name__ == '__main__':
    main()
