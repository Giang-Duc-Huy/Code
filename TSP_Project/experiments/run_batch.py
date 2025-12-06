"""
Batch experiment runner for TSP algorithms (Hill Climbing vs ACO).

Outputs per experiment:
- CSV summary with mean/std per algorithm per problem size
- JSON with raw per-run results
- Plots: mean±std bar charts and boxplots

Usage (example):
    python experiments\run_batch.py --sizes 10 30 50 --runs 5 --outdir experiments/results

"""
import argparse
import os
import json
import csv
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Ensure project root is on sys.path so imports like `comparison` work when
# this script is executed from the `experiments` folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from comparison import AlgorithmComparison
from utils.tsp_problem import TSProblem


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def summarize_results(raw_results: List[Dict[str, Any]]):
    """Compute mean and std for distance, fitness, time from a list of result dicts."""
    distances = np.array([r['distance'] for r in raw_results], dtype=float)
    fitnesses = np.array([r.get('fitness', 0) for r in raw_results], dtype=float)
    times = np.array([r['time'] for r in raw_results], dtype=float)

    return {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0,
        'mean_fitness': float(np.mean(fitnesses)),
        'std_fitness': float(np.std(fitnesses, ddof=1)) if len(fitnesses) > 1 else 0.0,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
    }


def run_batch(sizes: List[int], runs: int, hc_config: Dict[str, Any], aco_config: Dict[str, Any], outdir: str):
    ensure_dir(outdir)

    summary_rows = []
    raw_all = {}

    for n in sizes:
        print(f"\n=== Running size n={n} ({runs} runs) ===")
        raw_all[n] = {'Hill Climbing': [], 'ACO': []}

        for run_idx in range(runs):
            print(f"Run {run_idx+1}/{runs} for n={n} ...")
            # create new TSP instance per run (randomized)
            tsp = TSProblem(num_cities=n)
            comp = AlgorithmComparison(tsp)

            # HC
            hc_start = time.time()
            hc_result = comp.run_hill_climbing(
                max_iterations=hc_config.get('max_iterations', 10000),
                max_restarts=hc_config.get('max_restarts', 10),
                verbose=False
            )
            hc_end = time.time()
            hc_result['time'] = hc_result.get('time', hc_end - hc_start)

            # ACO
            aco_start = time.time()
            aco_result = comp.run_aco(
                num_ants=aco_config.get('num_ants', 50),
                num_iterations=aco_config.get('num_iterations', 100),
                alpha=aco_config.get('alpha', 1.0),
                beta=aco_config.get('beta', 5.0),
                evaporation=aco_config.get('evaporation', 0.5),
                verbose=False
            )
            aco_end = time.time()
            aco_result['time'] = aco_result.get('time', aco_end - aco_start)

            # Store raw
            raw_all[n]['Hill Climbing'].append(hc_result)
            raw_all[n]['ACO'].append(aco_result)

        # Summarize
        hc_summary = summarize_results(raw_all[n]['Hill Climbing'])
        aco_summary = summarize_results(raw_all[n]['ACO'])

        # Append summary rows
        summary_rows.append({
            'n': n,
            'algorithm': 'Hill Climbing',
            **hc_summary
        })
        summary_rows.append({
            'n': n,
            'algorithm': 'ACO',
            **aco_summary
        })

        # Save per-size raw json
        with open(os.path.join(outdir, f'raw_results_n{n}.json'), 'w', encoding='utf-8') as f:
            json.dump(raw_all[n], f, indent=2, ensure_ascii=False)

    # Save overall summary CSV
    csv_path = os.path.join(outdir, 'summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['n', 'algorithm', 'mean_distance', 'std_distance', 'mean_fitness', 'std_fitness', 'mean_time', 'std_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Save overall raw json
    with open(os.path.join(outdir, 'all_raw_results.json'), 'w', encoding='utf-8') as f:
        json.dump(raw_all, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary CSV to: {csv_path}")

    # Plot aggregated results
    plot_aggregates(summary_rows, outdir)
    plot_boxplots(raw_all, outdir)

    print('\nBatch run finished.')
    return outdir


def plot_aggregates(summary_rows: List[Dict[str, Any]], outdir: str):
    # Convert to structured dict by n
    grouped = defaultdict(dict)
    sizes = sorted(list({r['n'] for r in summary_rows}))
    for r in summary_rows:
        grouped[r['n']][r['algorithm']] = r

    # Distance mean bar with errorbars
    algs = ['Hill Climbing', 'ACO']
    x = np.arange(len(sizes))
    width = 0.35

    hc_means = [grouped[n]['Hill Climbing']['mean_distance'] for n in sizes]
    hc_stds = [grouped[n]['Hill Climbing']['std_distance'] for n in sizes]
    aco_means = [grouped[n]['ACO']['mean_distance'] for n in sizes]
    aco_stds = [grouped[n]['ACO']['std_distance'] for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, hc_means, width, yerr=hc_stds, label='Hill Climbing', color='#3498db', capsize=5)
    ax.bar(x + width/2, aco_means, width, yerr=aco_stds, label='ACO', color='#2ecc71', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel('Number of cities (n)')
    ax.set_ylabel('Distance (mean ± std)')
    ax.set_title('Mean Distance Comparison')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p = os.path.join(outdir, 'distance_mean_comparison.png')
    plt.savefig(p, dpi=200)
    print(f"Saved plot: {p}")
    plt.close(fig)

    # Time mean bar
    hc_means_t = [grouped[n]['Hill Climbing']['mean_time'] for n in sizes]
    hc_stds_t = [grouped[n]['Hill Climbing']['std_time'] for n in sizes]
    aco_means_t = [grouped[n]['ACO']['mean_time'] for n in sizes]
    aco_stds_t = [grouped[n]['ACO']['std_time'] for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, hc_means_t, width, yerr=hc_stds_t, label='Hill Climbing', color='#3498db', capsize=5)
    ax.bar(x + width/2, aco_means_t, width, yerr=aco_stds_t, label='ACO', color='#2ecc71', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel('Number of cities (n)')
    ax.set_ylabel('Time (s, mean ± std)')
    ax.set_title('Mean Time Comparison')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p = os.path.join(outdir, 'time_mean_comparison.png')
    plt.savefig(p, dpi=200)
    print(f"Saved plot: {p}")
    plt.close(fig)

    # Fitness mean bar
    hc_means_f = [grouped[n]['Hill Climbing']['mean_fitness'] for n in sizes]
    hc_stds_f = [grouped[n]['Hill Climbing']['std_fitness'] for n in sizes]
    aco_means_f = [grouped[n]['ACO']['mean_fitness'] for n in sizes]
    aco_stds_f = [grouped[n]['ACO']['std_fitness'] for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, hc_means_f, width, yerr=hc_stds_f, label='Hill Climbing', color='#3498db', capsize=5)
    ax.bar(x + width/2, aco_means_f, width, yerr=aco_stds_f, label='ACO', color='#2ecc71', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel('Number of cities (n)')
    ax.set_ylabel('Fitness (mean ± std)')
    ax.set_title('Mean Fitness Comparison')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p = os.path.join(outdir, 'fitness_mean_comparison.png')
    plt.savefig(p, dpi=200)
    print(f"Saved plot: {p}")
    plt.close(fig)


def plot_boxplots(raw_all: Dict[int, Dict[str, List[Dict[str, Any]]]], outdir: str):
    # For each size, create a boxplot comparing distances and fitness
    for n, data in raw_all.items():
        hc_dist = [r['distance'] for r in data['Hill Climbing']]
        aco_dist = [r['distance'] for r in data['ACO']]
        hc_fitness = [r.get('fitness', 0) for r in data['Hill Climbing']]
        aco_fitness = [r.get('fitness', 0) for r in data['ACO']]

        # Distance boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([hc_dist, aco_dist], tick_labels=['HC', 'ACO'], notch=True)
        ax.set_title(f'Distance distribution (n={n})')
        ax.set_ylabel('Distance')
        plt.tight_layout()
        p = os.path.join(outdir, f'boxplot_distance_n{n}.png')
        plt.savefig(p, dpi=200)
        print(f"Saved plot: {p}")
        plt.close(fig)

        # Fitness boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([hc_fitness, aco_fitness], tick_labels=['HC', 'ACO'], notch=True)
        ax.set_title(f'Fitness distribution (n={n})')
        ax.set_ylabel('Fitness')
        plt.tight_layout()
        p = os.path.join(outdir, f'boxplot_fitness_n{n}.png')
        plt.savefig(p, dpi=200)
        print(f"Saved plot: {p}")
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch experiments for TSP algorithms')
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 30, 50], help='Problem sizes (list of n)')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per size')
    parser.add_argument('--hc_iterations', type=int, default=10000)
    parser.add_argument('--hc_restarts', type=int, default=10)
    parser.add_argument('--aco_ants', type=int, default=50)
    parser.add_argument('--aco_iterations', type=int, default=100)
    parser.add_argument('--outdir', type=str, default='experiments/results', help='Output directory')

    args = parser.parse_args()

    hc_cfg = {'max_iterations': args.hc_iterations, 'max_restarts': args.hc_restarts}
    aco_cfg = {'num_ants': args.aco_ants, 'num_iterations': args.aco_iterations}

    out = run_batch(args.sizes, args.runs, hc_cfg, aco_cfg, args.outdir)
    print(f"Results saved under: {out}")
