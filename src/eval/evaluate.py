#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import pandas as pd
from scipy.spatial.distance import jensenshannon as jsd
from sklearn.metrics import precision_score, recall_score
import jismesh.utils as ju
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from fastdtw import fastdtw
import numpy as np
from scipy.interpolate import interp1d

# 将项目根目录添加到 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# =============================================================================
# 工具类与函数
# =============================================================================
class DictToObject:
    """递归将字典转换为对象，方便通过属性方式访问配置参数。"""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, DictToObject(v))
            else:
                setattr(self, k, v)


def get_boundary(df):
    """返回轨迹 DataFrame 的经纬度边界"""
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    return {'lati_min': lat_min, 'lati_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max}


def divide_grids(boundary, grid_num):
    """将边界按等间距分割成 grid_num 个区间，返回纬度和经度分割数组"""
    lat_interval = (boundary['lati_max'] - boundary['lati_min']) / grid_num
    lon_interval = (boundary['lon_max'] - boundary['lon_min']) / grid_num
    lat_grids = np.arange(boundary['lati_min'], boundary['lati_max'], lat_interval)
    lon_grids = np.arange(boundary['lon_min'], boundary['lon_max'], lon_interval)
    return lat_grids, lon_grids


def latlon2grid(lat, lon, lat_grids, lon_grids):
    """将经纬度转换为网格索引"""
    lat_idx = np.searchsorted(lat_grids, lat) - 1
    lon_idx = np.searchsorted(lon_grids, lon) - 1
    return lat_idx, lon_idx


# 用于还原 head 中各字段名称（若需要）
switcher = {
    0: 'departure_time',
    1: 'total_dis',
    2: 'total_time',
    3: 'total_len',
    4: 'avg_dis',
    5: 'avg_speed',
    6: 'starting_location',
    7: 'ending_location'
}


# 共同的小工具：按 uid 提取轨迹 (lat, lon) 序列
def _traj_by_uid(df, uid):
    d = df[df['uid'] == uid].sort_values('time')
    coords = list(zip(d['latitude'].values, d['longitude'].values))
    return coords if len(coords) > 1 else None

def _summarize(dist_list):
    arr = np.asarray(dist_list, dtype=float)
    return {
        'n_pairs': int(arr.size),
        'mean': float(arr.mean()) if arr.size else float('nan'),
        'std': float(arr.std(ddof=1)) if arr.size > 1 else float('nan'),
        'median': float(np.median(arr)) if arr.size else float('nan'),
        'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)) if arr.size else float('nan'),
        'p10': float(np.percentile(arr, 10)) if arr.size else float('nan'),
        'p90': float(np.percentile(arr, 90)) if arr.size else float('nan'),
    }

def _choose_uids(gt_df, gen_df, sample_size=None, random_state=42):
    # 只取真实和生成的 uid 交集，确保一一对应
    uids = np.intersect1d(gt_df['uid'].unique(), gen_df['uid'].unique())
    # 固定顺序，保证可复现
    uids = np.array(sorted(uids))
    if sample_size is not None and sample_size < len(uids):
        rng = np.random.default_rng(random_state)
        uids = rng.choice(uids, size=sample_size, replace=False)
        uids = np.array(sorted(uids))
    return uids

def summarize_indicator(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return dict(indicator=np.nan, median=np.nan, iqr=np.nan,
                    mean=np.nan, std=np.nan, p10=np.nan, p90=np.nan)
    return {
        # pick ONE to report as the paper's indicator (I suggest median)
        "indicator": float(np.median(v)),
        # extras you might log
        "median": float(np.median(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }
def geo_distance_m(p, q):
    EARTH_RADIUS_M = 6371000.0  # 地球半径（米）
    """
    Haversine 大圆距离（单位：米）
    p, q: (lat, lon)，单位：度
    """
    lat1, lon1 = np.radians(p[0]), np.radians(p[1])
    lat2, lon2 = np.radians(q[0]), np.radians(q[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(a))

# 1) 按 uid 的 DTW
def calculate_dtw_distance(gt_df, gen_df, sample_size=None, radius=1, random_state=42):
    """
    按 uid 一一对应计算 DTW 距离（欧氏距离；不做归一化）。
    - radius: fastdtw 的半径，建议 1 或 2，可显著减少过度扭曲
    """
    uids = _choose_uids(gt_df, gen_df, sample_size, random_state)
    dtw_distances = []
    skipped = 0

    for uid in uids:
        gt = _traj_by_uid(gt_df, uid)
        ge = _traj_by_uid(gen_df, uid)
        if gt is None or ge is None:
            skipped += 1
            continue
        dist, _ = fastdtw(gt, ge, dist=geo_distance_m, radius=radius)
        dtw_distances.append(dist)

    stats = _summarize(dtw_distances)
    stats['skipped_pairs'] = skipped
    return stats, dtw_distances

# 2) 按 uid 的 Fréchet
def calculate_frechet_distance(gt_df, gen_df, sample_size=None, target_cap=50, random_state=42):
    """
    按 uid 一一对应计算离散 Fréchet 距离。
    - 仍采用线性插值到 target_length = min(len(gt), len(gen), target_cap)
    - 不做归一化
    """
    def frechet_distance(P, Q):
        P = np.asarray(P); Q = np.asarray(Q)
        ca = np.full((len(P), len(Q)), -1.0)
        def _c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            if i == 0 and j == 0:
                ca[i, j] = geo_distance_m(P[0], Q[0])
            elif i > 0 and j == 0:
                ca[i, j] = max(_c(i-1, 0), geo_distance_m(P[i], Q[0]))
            elif i == 0 and j > 0:
                ca[i, j] = max(_c(0, j-1), geo_distance_m(P[0], Q[j]))
            else:
                ca[i, j] = max(min(_c(i-1, j), _c(i-1, j-1), _c(i, j-1)), geo_distance_m(P[i], Q[j]))
            return ca[i, j]
        return _c(len(P)-1, len(Q)-1)

    def resample_linear(curve, target_len):
        curve = np.asarray(curve)
        if len(curve) == target_len:
            return curve
        t_old = np.linspace(0, 1, len(curve))
        t_new = np.linspace(0, 1, target_len)
        x = interp1d(t_old, curve[:, 0], kind='linear')(t_new)
        y = interp1d(t_old, curve[:, 1], kind='linear')(t_new)
        return np.column_stack([x, y])

    uids = _choose_uids(gt_df, gen_df, sample_size, random_state)
    frechet_distances = []
    skipped = 0

    for uid in uids:
        gt = _traj_by_uid(gt_df, uid)
        ge = _traj_by_uid(gen_df, uid)
        if gt is None or ge is None:
            skipped += 1
            continue

        gt = np.asarray(gt); ge = np.asarray(ge)
        target_len = int(min(len(gt), len(ge), target_cap))
        if target_len < 2:
            skipped += 1
            continue

        gt_r = resample_linear(gt, target_len)
        ge_r = resample_linear(ge, target_len)
        dist = frechet_distance(gt_r, ge_r)
        frechet_distances.append(dist)

    stats = _summarize(frechet_distances)
    stats['skipped_pairs'] = skipped
    return stats, frechet_distances

# 3) 分位数曲线绘图（可选）
def plot_quantile_curve(values, title='DTW quantiles', save_path=None):
    import matplotlib.pyplot as plt
    v = np.asarray(values, dtype=float)
    qs = np.linspace(0, 1, 101)
    qv = np.quantile(v, qs)
    plt.figure(figsize=(5,3.2))
    plt.plot(qs, qv)
    plt.xlabel('quantile')
    plt.ylabel('distance (m)')  # 明确单位：米
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    return qs, qv

# =============================================================================
# 可视化函数
# =============================================================================
def create_mesh_size_analysis(df_summary, summary_dir, mesh_sizes):
    """Create analysis comparing different mesh sizes"""

    print(f"\n{'=' * 80}")
    print("MESH SIZE COMPARISON ANALYSIS")
    print(f"{'=' * 80}")

    # Group by mesh size and calculate mean metrics
    mesh_comparison = df_summary.groupby('mesh_size_param').agg({
        'length_error': ['mean', 'std'],
        'pattern_score': ['mean', 'std'],
        'trip_js': ['mean', 'std'],
        'density_js': ['mean', 'std']
    }).round(6)

    print("\nMean ± Std for each mesh size:")
    print("-" * 80)
    for mesh_size in mesh_sizes:
        mesh_data = mesh_comparison.loc[mesh_size]
        print(f"\nMesh Size {mesh_size}:")
        print(f"  Length Error:  {mesh_data[('length_error', 'mean')]:.6f} ± {mesh_data[('length_error', 'std')]:.6f}")
        print(
            f"  Pattern Score: {mesh_data[('pattern_score', 'mean')]:.6f} ± {mesh_data[('pattern_score', 'std')]:.6f}")
        print(f"  Trip JS:       {mesh_data[('trip_js', 'mean')]:.6f} ± {mesh_data[('trip_js', 'std')]:.6f}")
        print(f"  Density JS:    {mesh_data[('density_js', 'mean')]:.6f} ± {mesh_data[('density_js', 'std')]:.6f}")

    # Save detailed mesh comparison
    mesh_comparison.to_csv(os.path.join(summary_dir, 'mesh_size_comparison.csv'))

    # Create visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['length_error', 'pattern_score', 'trip_js', 'density_js']
        titles = ['Length Error', 'Pattern Score', 'Trip JS Divergence', 'Density JS Divergence']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]

            means = [mesh_comparison.loc[size, (metric, 'mean')] for size in mesh_sizes]
            stds = [mesh_comparison.loc[size, (metric, 'std')] for size in mesh_sizes]

            ax.errorbar(mesh_sizes, means, yerr=stds, marker='o', capsize=5, capthick=2)
            ax.set_xlabel('Mesh Size')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Mesh Size')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'mesh_size_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError:
        print("Matplotlib not available, skipping mesh size comparison plot")

    print(f"\nMesh size analysis saved to: {summary_dir}")
def create_comprehensive_visualizations(gt_df, gen_df, vis_dir, args):
    """Create comprehensive visualizations for trajectory evaluation"""

    # 1. Spatial Distribution Comparison
    create_spatial_distribution_plots(gt_df, gen_df, vis_dir)

    # 2. Trajectory Length Distribution
    create_length_distribution_plots(gt_df, gen_df, vis_dir)

    # 3. Start/End Point Analysis
    create_start_end_analysis(gt_df, gen_df, vis_dir, args)

    # 4. Density Heatmaps
    create_density_heatmaps(gt_df, gen_df, vis_dir)

    # 5. Trajectory Samples Visualization
    create_trajectory_samples(gt_df, gen_df, vis_dir)

    # 6. Statistical Comparison
    create_statistical_comparison(gt_df, gen_df, vis_dir)

    # 7. Interactive Visualizations
    create_interactive_visualizations(gt_df, gen_df, vis_dir)

    print(f"All visualizations saved to: {vis_dir}")


def create_spatial_distribution_plots(gt_df, gen_df, vis_dir, alpha_scatter=0.05, alpha_overlay=0.15):
    """Create spatial distribution comparison plots with adjustable transparency"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Get ground truth boundaries
    lon_min, lon_max = gt_df['longitude'].min(), gt_df['longitude'].max()
    lat_min, lat_max = gt_df['latitude'].min(), gt_df['latitude'].max()

    # Add small padding for better visualization
    lon_padding = (lon_max - lon_min) * 0.02
    lat_padding = (lat_max - lat_min) * 0.02

    # Filter generated data to GT range
    gen_filtered = gen_df[
        (gen_df['longitude'] >= lon_min) & (gen_df['longitude'] <= lon_max) &
        (gen_df['latitude'] >= lat_min) & (gen_df['latitude'] <= lat_max)
    ]

    # Ground truth scatter
    axes[0, 0].scatter(gt_df['longitude'], gt_df['latitude'],
                       alpha=alpha_scatter, s=0.5, c='blue', rasterized=True)
    axes[0, 0].set_title('Ground Truth Spatial Distribution')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axes[0, 0].set_ylim(lat_min - lat_padding, lat_max + lat_padding)

    # Generated scatter
    axes[0, 1].scatter(gen_filtered['longitude'], gen_filtered['latitude'],
                       alpha=alpha_scatter, s=0.5, c='red', rasterized=True)
    axes[0, 1].set_title('Generated Spatial Distribution')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axes[0, 1].set_ylim(lat_min - lat_padding, lat_max + lat_padding)

    # Overlay comparison - plot generated first, then ground truth
    axes[1, 0].scatter(gen_filtered['longitude'], gen_filtered['latitude'],
                       alpha=alpha_overlay, s=0.5, c='red', label='Generated',
                       rasterized=True, zorder=1)
    axes[1, 0].scatter(gt_df['longitude'], gt_df['latitude'],
                       alpha=alpha_overlay, s=0.5, c='blue', label='Ground Truth',
                       rasterized=True, zorder=2)
    axes[1, 0].set_title('Overlay Comparison')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axes[1, 0].set_ylim(lat_min - lat_padding, lat_max + lat_padding)

    # Difference in density (using GT boundaries)
    gt_hist, xedges, yedges = np.histogram2d(
        gt_df['longitude'], gt_df['latitude'],
        bins=50,
        range=[[lon_min, lon_max], [lat_min, lat_max]]
    )
    gen_hist, _, _ = np.histogram2d(
        gen_filtered['longitude'], gen_filtered['latitude'],
        bins=[xedges, yedges]
    )
    diff = gt_hist - gen_hist
    im = axes[1, 1].imshow(diff.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('Density Difference (GT - Generated)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_xlim(lon_min - lon_padding, lon_max + lon_padding)
    axes[1, 1].set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spatial_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
def create_length_distribution_plots(gt_df, gen_df, vis_dir):
    """Create trajectory length distribution plots"""
    gt_lengths = gt_df.groupby('uid').apply(calculate_travel_distance)
    gen_lengths = gen_df.groupby('uid').apply(calculate_travel_distance)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Histograms
    axes[0, 0].hist(gt_lengths, bins=50, alpha=0.7, label='Ground Truth', color='blue', density=True)
    axes[0, 0].hist(gen_lengths, bins=50, alpha=0.7, label='Generated', color='red', density=True)
    axes[0, 0].set_title('Trajectory Length Distribution')
    axes[0, 0].set_xlabel('Length')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # Box plots
    data_to_plot = [gt_lengths, gen_lengths]
    axes[0, 1].boxplot(data_to_plot, tick_labels=['Ground Truth', 'Generated'])
    axes[0, 1].set_title('Trajectory Length Box Plot')
    axes[0, 1].set_ylabel('Length')

    # Q-Q plot
    from scipy import stats
    gt_sorted = np.sort(gt_lengths)
    gen_sorted = np.sort(gen_lengths)
    min_len = min(len(gt_sorted), len(gen_sorted))
    gt_quantiles = gt_sorted[np.linspace(0, len(gt_sorted) - 1, min_len).astype(int)]
    gen_quantiles = gen_sorted[np.linspace(0, len(gen_sorted) - 1, min_len).astype(int)]

    axes[1, 0].scatter(gt_quantiles, gen_quantiles, alpha=0.6)
    min_val = min(gt_quantiles.min(), gen_quantiles.min())
    max_val = max(gt_quantiles.max(), gen_quantiles.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 0].set_xlabel('Ground Truth Quantiles')
    axes[1, 0].set_ylabel('Generated Quantiles')
    axes[1, 0].set_title('Q-Q Plot: Length Distribution')

    # Cumulative distribution
    axes[1, 1].hist(gt_lengths, bins=50, alpha=0.7, label='Ground Truth', color='blue',
                    density=True, cumulative=True, histtype='step', linewidth=2)
    axes[1, 1].hist(gen_lengths, bins=50, alpha=0.7, label='Generated', color='red',
                    density=True, cumulative=True, histtype='step', linewidth=2)
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Length')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'length_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_start_end_analysis(gt_df, gen_df, vis_dir, args):
    """Create start and end point analysis"""
    # Get start and end points for each trajectory
    gt_start = gt_df.groupby('uid').first()[['latitude', 'longitude']]
    gt_end = gt_df.groupby('uid').last()[['latitude', 'longitude']]
    gen_start = gen_df.groupby('uid').first()[['latitude', 'longitude']]
    gen_end = gen_df.groupby('uid').last()[['latitude', 'longitude']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Start points
    axes[0, 0].scatter(gt_start['longitude'], gt_start['latitude'], alpha=0.6, s=20, c='blue', label='GT Start')
    axes[0, 0].set_title('Ground Truth Start Points')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')

    axes[0, 1].scatter(gen_start['longitude'], gen_start['latitude'], alpha=0.6, s=20, c='red', label='Gen Start')
    axes[0, 1].set_title('Generated Start Points')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')

    axes[0, 2].scatter(gt_start['longitude'], gt_start['latitude'], alpha=0.6, s=20, c='blue', label='GT Start')
    axes[0, 2].scatter(gen_start['longitude'], gen_start['latitude'], alpha=0.6, s=20, c='red', label='Gen Start')
    axes[0, 2].set_title('Start Points Comparison')
    axes[0, 2].set_xlabel('Longitude')
    axes[0, 2].set_ylabel('Latitude')
    axes[0, 2].legend()

    # End points
    axes[1, 0].scatter(gt_end['longitude'], gt_end['latitude'], alpha=0.6, s=20, c='blue', label='GT End')
    axes[1, 0].set_title('Ground Truth End Points')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')

    axes[1, 1].scatter(gen_end['longitude'], gen_end['latitude'], alpha=0.6, s=20, c='red', label='Gen End')
    axes[1, 1].set_title('Generated End Points')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')

    axes[1, 2].scatter(gt_end['longitude'], gt_end['latitude'], alpha=0.6, s=20, c='blue', label='GT End')
    axes[1, 2].scatter(gen_end['longitude'], gen_end['latitude'], alpha=0.6, s=20, c='red', label='Gen End')
    axes[1, 2].set_title('End Points Comparison')
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'start_end_points_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_density_heatmaps(gt_df, gen_df, vis_dir):
    """Create density heatmaps"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground truth heatmap
    gt_hist, xedges, yedges = np.histogram2d(gt_df['longitude'], gt_df['latitude'], bins=50)
    gt_hist_norm = gt_hist / gt_hist.sum()  # Normalize to 0-1
    im1 = axes[0].imshow(gt_hist_norm.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth Density')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0])

    # Generated heatmap
    gen_hist, _, _ = np.histogram2d(gen_df['longitude'], gen_df['latitude'], bins=[xedges, yedges])
    gen_hist_norm = gen_hist / gen_hist.sum()  # Normalize to 0-1
    im2 = axes[1].imshow(gen_hist_norm.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap='Reds', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Generated Density')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1])

    # Difference heatmap
    diff = gt_hist_norm - gen_hist_norm
    max_abs_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap='RdBu_r', aspect='auto', vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title('Density Difference (GT - Generated)')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'density_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_trajectory_samples(gt_df, gen_df, vis_dir, n_samples=10):
    """Visualize sample trajectories"""
    gt_uids = gt_df['uid'].unique()[:n_samples]
    gen_uids = gen_df['uid'].unique()[:n_samples]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, uid in enumerate(gt_uids):
        if i >= 10: break
        traj = gt_df[gt_df['uid'] == uid].sort_values('time')
        axes[i].plot(traj['longitude'], traj['latitude'], 'b-', linewidth=2, alpha=0.7)
        axes[i].scatter(traj['longitude'].iloc[0], traj['latitude'].iloc[0], c='green', s=50, marker='o', label='Start')
        axes[i].scatter(traj['longitude'].iloc[-1], traj['latitude'].iloc[-1], c='red', s=50, marker='s', label='End')
        axes[i].set_title(f'GT Trajectory {uid}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'ground_truth_trajectory_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, uid in enumerate(gen_uids):
        if i >= 10: break
        traj = gen_df[gen_df['uid'] == uid].sort_values('time')
        axes[i].plot(traj['longitude'], traj['latitude'], 'r-', linewidth=2, alpha=0.7)
        axes[i].scatter(traj['longitude'].iloc[0], traj['latitude'].iloc[0], c='green', s=50, marker='o', label='Start')
        axes[i].scatter(traj['longitude'].iloc[-1], traj['latitude'].iloc[-1], c='red', s=50, marker='s', label='End')
        axes[i].set_title(f'Generated Trajectory {uid}')
        axes[i].set_xlabel('Longitude')
        axes[i].set_ylabel('Latitude')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'generated_trajectory_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_statistical_comparison(gt_df, gen_df, vis_dir):
    """Create statistical comparison plots"""
    # Calculate statistics
    gt_stats = {
        'lat_mean': gt_df['latitude'].mean(),
        'lat_std': gt_df['latitude'].std(),
        'lon_mean': gt_df['longitude'].mean(),
        'lon_std': gt_df['longitude'].std(),
        'n_trajectories': gt_df['uid'].nunique(),
        'n_points': len(gt_df)
    }

    gen_stats = {
        'lat_mean': gen_df['latitude'].mean(),
        'lat_std': gen_df['latitude'].std(),
        'lon_mean': gen_df['longitude'].mean(),
        'lon_std': gen_df['longitude'].std(),
        'n_trajectories': gen_df['uid'].nunique(),
        'n_points': len(gen_df)
    }

    # Create comparison bar chart
    metrics = list(gt_stats.keys())
    gt_values = list(gt_stats.values())
    gen_values = list(gen_stats.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, gt_values, width, label='Ground Truth', alpha=0.8)
    bars2 = ax.bar(x + width / 2, gen_values, width, label='Generated', alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Statistical Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_interactive_visualizations(gt_df, gen_df, vis_dir):
    """Create interactive visualizations using plotly"""
    try:
        # Sample data for performance
        gt_sample = gt_df.sample(min(5000, len(gt_df)))
        gen_sample = gen_df.sample(min(5000, len(gen_df)))

        # Interactive scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=gt_sample['longitude'],
            y=gt_sample['latitude'],
            mode='markers',
            name='Ground Truth',
            marker=dict(size=3, color='blue', opacity=0.6)
        ))

        fig.add_trace(go.Scattergl(
            x=gen_sample['longitude'],
            y=gen_sample['latitude'],
            mode='markers',
            name='Generated',
            marker=dict(size=3, color='red', opacity=0.6)
        ))

        fig.update_layout(
            title='Interactive Trajectory Points Comparison',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            hovermode='closest'
        )

        fig.write_html(os.path.join(vis_dir, 'interactive_scatter.html'))

        # Interactive trajectory samples
        fig_traj = go.Figure()

        sample_uids_gt = gt_df['uid'].unique()[:5]
        sample_uids_gen = gen_df['uid'].unique()[:5]

        for i, uid in enumerate(sample_uids_gt):
            traj = gt_df[gt_df['uid'] == uid].sort_values('time')
            fig_traj.add_trace(go.Scattergl(
                x=traj['longitude'],
                y=traj['latitude'],
                mode='lines+markers',
                name=f'GT Traj {uid}',
                line=dict(color=f'blue', width=2),
                marker=dict(size=4)
            ))

        for i, uid in enumerate(sample_uids_gen):
            traj = gen_df[gen_df['uid'] == uid].sort_values('time')
            fig_traj.add_trace(go.Scattergl(
                x=traj['longitude'],
                y=traj['latitude'],
                mode='lines+markers',
                name=f'Gen Traj {uid}',
                line=dict(color=f'red', width=2),
                marker=dict(size=4)
            ))

        fig_traj.update_layout(
            title='Sample Trajectories Comparison',
            xaxis_title='Longitude',
            yaxis_title='Latitude'
        )

        fig_traj.write_html(os.path.join(vis_dir, 'interactive_trajectories.html'))

    except Exception as e:
        print(f"Could not create interactive visualizations: {e}")

# =============================================================================
# 评价指标计算函数
# =============================================================================
def calculate_density_error(gt_df, gen_df, consider_hour=False, division_type='DIVISION', mesh_degree=3, grid_num=16):
    """
    计算轨迹点密度分布的 JS 散度。
    """
    if division_type.upper() == 'JISMESH':
        gt_df['meshcode'] = gt_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                        axis=1)
        # to debug: before applying to_meshcode, resrict the latitude and longitude within the boundary
        gen_df['latitude'] = gen_df['latitude'].clip(gt_df['latitude'].min(), gt_df['latitude'].max())
        gen_df['longitude'] = gen_df['longitude'].clip(gt_df['longitude'].min(), gt_df['longitude'].max())
        gen_df['meshcode'] = gen_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                          axis=1)
    else:
        boundary = get_boundary(gt_df)
        lat_grids, lon_grids = divide_grids(boundary, grid_num)
        gt_df['meshcode'] = gt_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)
        gen_df['meshcode'] = gen_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)

    if consider_hour:
        gt_df['hour'] = pd.to_datetime(gt_df['time']).dt.hour
        gen_df['hour'] = pd.to_datetime(gen_df['time']).dt.hour
        count_gt = gt_df.groupby(['hour', 'meshcode'])['uid'].count()
        count_gen = gen_df.groupby(['hour', 'meshcode'])['uid'].count()
        merged = pd.merge(count_gt, count_gen, on=['hour', 'meshcode'], how='outer', suffixes=('_gt', '_gen')).fillna(0)
        return jsd(merged['uid_gt'].values, merged['uid_gen'].values)
    else:
        count_gt = gt_df['meshcode'].value_counts()
        count_gen = gen_df['meshcode'].value_counts()
        merged = pd.merge(count_gt, count_gen, left_index=True, right_index=True, how='outer',
                          suffixes=('_gt', '_gen')).fillna(0)
        return jsd(merged.iloc[:, 0].values, merged.iloc[:, 1].values)


def calculate_trip_error(gt_df, gen_df, division_type='DIVISION', mesh_degree=3, grid_num=16):
    """
    计算轨迹起点与终点分布的 JS 散度。
    """
    if division_type.upper() == 'JISMESH':
        gt_df['meshcode'] = gt_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                        axis=1)
        # to debug: before applying to_meshcode, resrict the latitude and longitude within the boundary
        gen_df['latitude'] = gen_df['latitude'].clip(gt_df['latitude'].min(), gt_df['latitude'].max())
        gen_df['longitude'] = gen_df['longitude'].clip(gt_df['longitude'].min(), gt_df['longitude'].max())
        gen_df['meshcode'] = gen_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                          axis=1)
    else:
        boundary = get_boundary(gt_df)
        lat_grids, lon_grids = divide_grids(boundary, grid_num)
        gt_df['meshcode'] = gt_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)
        gen_df['meshcode'] = gen_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)

    gt_df['start_mesh'] = gt_df.groupby('uid')['meshcode'].transform('first')
    gt_df['end_mesh'] = gt_df.groupby('uid')['meshcode'].transform('last')
    gen_df['start_mesh'] = gen_df.groupby('uid')['meshcode'].transform('first')
    gen_df['end_mesh'] = gen_df.groupby('uid')['meshcode'].transform('last')
    gt_df = gt_df.drop_duplicates(subset=['uid'])
    gen_df = gen_df.drop_duplicates(subset=['uid'])

    start_count_gt = gt_df['start_mesh'].value_counts()
    start_count_gen = gen_df['start_mesh'].value_counts()
    merged_start = pd.merge(start_count_gt, start_count_gen, left_index=True, right_index=True, how='outer',
                            suffixes=('_gt', '_gen')).fillna(0)
    end_count_gt = gt_df['end_mesh'].value_counts()
    end_count_gen = gen_df['end_mesh'].value_counts()
    merged_end = pd.merge(end_count_gt, end_count_gen, left_index=True, right_index=True, how='outer',
                          suffixes=('_gt', '_gen')).fillna(0)

    start_js = jsd(merged_start.iloc[:, 0].values, merged_start.iloc[:, 1].values)
    end_js = jsd(merged_end.iloc[:, 0].values, merged_end.iloc[:, 1].values)
    return (start_js + end_js) / 2


def calcu_pattern_score(gt_df, gen_df, top_n=100, division_type='DIVISION', mesh_degree=3, grid_num=16):
    """
    计算热门区域的 F1 分数。
    """
    from collections import Counter
    def get_top_n_meshcodes(df, n):
        counts = Counter(df['meshcode'])
        return [code for code, _ in counts.most_common(n)]

    if division_type.upper() == 'JISMESH':
        gt_df['meshcode'] = gt_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                        axis=1)
        # to debug: before applying to_meshcode, resrict the latitude and longitude within the boundary
        gen_df['latitude'] = gen_df['latitude'].clip(gt_df['latitude'].min(), gt_df['latitude'].max())
        gen_df['longitude'] = gen_df['longitude'].clip(gt_df['longitude'].min(), gt_df['longitude'].max())
        gen_df['meshcode'] = gen_df.apply(lambda row: ju.to_meshcode(row['latitude'], row['longitude'], mesh_degree),
                                          axis=1)
    else:
        boundary = get_boundary(gt_df)
        lat_grids, lon_grids = divide_grids(boundary, grid_num)
        gt_df['meshcode'] = gt_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)
        gen_df['meshcode'] = gen_df.apply(
            lambda row: latlon2grid(row['latitude'], row['longitude'], lat_grids, lon_grids), axis=1)

    top_gt = get_top_n_meshcodes(gt_df, top_n)
    top_gen = get_top_n_meshcodes(gen_df, top_n)
    combined = set(top_gt).union(set(top_gen))

    def binary_vector(mesh_list, all_codes):
        return [1 if code in mesh_list else 0 for code in all_codes]

    vec_gt = binary_vector(top_gt, combined)
    vec_gen = binary_vector(top_gen, combined)
    prec = precision_score(vec_gt, vec_gen, average='micro')
    rec = recall_score(vec_gt, vec_gen, average='micro')
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)


def calculate_travel_distance(df):
    """计算单条轨迹的欧几里得距离和（按时间顺序排序）。"""
    df_sorted = df.sort_values(by='time')
    lats = df_sorted['latitude'].values
    lons = df_sorted['longitude'].values
    distances = np.sqrt(np.diff(lats) ** 2 + np.diff(lons) ** 2)
    return np.sum(distances)


def length_error(gt_df, gen_df):
    """计算真实轨迹与生成轨迹平均行程长度的绝对误差。"""
    gt_dist = gt_df.groupby('uid').apply(calculate_travel_distance)
    gen_dist = gen_df.groupby('uid').apply(calculate_travel_distance)
    return np.abs(gt_dist.mean() - gen_dist.mean())

def length_relative_error(gt_df, gen_df):
    """计算真实轨迹与生成轨迹平均行程长度的相对误差。"""
    gt_dist = gt_df.groupby('uid').apply(calculate_travel_distance)
    gen_dist = gen_df.groupby('uid').apply(calculate_travel_distance)
    # 避免除零错误，只对真实距离大于0的轨迹计算相对误差
    valid_mask = gt_dist > 0
    if valid_mask.sum() == 0:
        return 0.0
    relative_errors = np.abs(gt_dist[valid_mask] - gen_dist[valid_mask]) / gt_dist[valid_mask]
    return np.mean(relative_errors)

# =============================================================================
# 主评价流程
# =============================================================================
# def evaluate_trajectories(time_str, condition_mode):
#     """
#     根据给定的时间字符串和条件模式查找对应的生成结果文件夹，
#     加载生成轨迹与真实轨迹数据，计算各项评价指标，并保存评价结果至 evaluation.csv。
#
#     参数：
#       time_str: 时间字符串（例如 "05-09-15-22-24"）
#       condition_mode: 条件模式 ("real" 或 "placeholder")
#     """
#     test_dir = "./TEST/"
#     matching_folder = None
#     # 查找文件夹名称中同时包含 time_str 与 condition_mode 的文件夹
#     for folder in os.listdir(test_dir):
#         if time_str in folder and condition_mode in folder:
#             matching_folder = os.path.join(test_dir, folder)
#             break
#
#     if matching_folder is None:
#         print(f"未找到包含时间字符串 {time_str} 和条件 {condition_mode} 的测试结果文件夹。")
#         sys.exit(1)
#
#     gen_csv = os.path.join(matching_folder, "generated_trajectories.csv")
#     gt_csv = os.path.join(matching_folder, "ground_truth_trajectories.csv")
#     if not os.path.exists(gen_csv) or not os.path.exists(gt_csv):
#         print("生成轨迹或真实轨迹文件不存在，请检查！")
#         sys.exit(1)
#
#     gt_df = pd.read_csv(gt_csv)
#     gen_df = pd.read_csv(gen_csv)
#
#     # 评价参数设置（此处默认使用 DIVISION 分割方式）
#     division_type = "DIVISION"
#     mesh_size = 500
#     grid_num = 16
#     if division_type.upper() == "JISMESH":
#         jismesh_switcher = {80000: 1, 40000: 40, 20000: 20, 16000: 16,
#                             10000: 2, 8000: 8, 5000: 5, 4000: 4, 2500: 2.5, 2000: 2,
#                             1000: 3, 500: 4, 250: 5, 125: 6}
#         mesh_degree = jismesh_switcher.get(mesh_size, 3)
#     else:
#         mesh_degree = -1
#
#     density_js = calculate_density_error(gt_df.copy(), gen_df.copy(),
#                                          consider_hour=False,
#                                          division_type=division_type,
#                                          mesh_degree=mesh_degree,
#                                          grid_num=grid_num)
#     trip_js = calculate_trip_error(gt_df.copy(), gen_df.copy(),
#                                    division_type=division_type,
#                                    mesh_degree=mesh_degree,
#                                    grid_num=grid_num)
#     pattern_score = calcu_pattern_score(gt_df.copy(), gen_df.copy(),
#                                         top_n=100,
#                                         division_type=division_type,
#                                         mesh_degree=mesh_degree,
#                                         grid_num=grid_num)
#     len_err = length_relative_error(gt_df, gen_df)
#
#     print("Evaluation for time_str =", time_str, "and condition =", condition_mode)
#     print("Length Error:", len_err)
#     print("Pattern Score:", pattern_score)
#     print("Trip JS Divergence:", trip_js)
#     print("Density JS Divergence:", density_js)
#
#     # 保存评价结果到 evaluation.csv（在当前目录下）
#     eval_csv = "evaluation.csv"
#     if os.path.exists(eval_csv):
#         eval_df = pd.read_csv(eval_csv)
#     else:
#         eval_df = pd.DataFrame(columns=["TIME_STR", "CONDITION", "DIVISION_TYPE", "MESH_SIZE",
#                                         "Length_Error", "Pattern_Score", "Trip_JS", "Density_JS"])
#     exists = ((eval_df["TIME_STR"] == time_str) &
#               (eval_df["MESH_SIZE"] == mesh_size) &
#               (eval_df["CONDITION"] == condition_mode)).any()
#     if exists:
#         print("该评价结果已存在！")
#     else:
#         new_record = {
#             "TIME_STR": time_str,
#             "CONDITION": condition_mode,
#             "DIVISION_TYPE": division_type,
#             "MESH_SIZE": mesh_size,
#             "Length_Error": f"{len_err:.4f}",
#             "Pattern_Score": f"{pattern_score:.4f}",
#             "Trip_JS": f"{trip_js:.4f}",
#             "Density_JS": f"{density_js:.4f}"
#         }
#         eval_df = eval_df.append(new_record, ignore_index=True)
#         eval_df.to_csv(eval_csv, index=False)
#         print("评价结果已保存到", eval_csv)


def load_config_info(folder_path):
    """Load configuration information from the generation folder"""
    config_file = os.path.join(folder_path, 'config.yaml')
    config_info = {}

    if os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Extract inference parameters correctly
            inference_config = config.get('inference', {})

            # Check if baseline is enabled and set model_type accordingly
            baseline_enabled = config.get('baseline', {}).get('enabled', False)
            if baseline_enabled:
                baseline_type = config.get('baseline', {}).get('type', 'unknown')
                model_type = baseline_type  # Use baseline type as model_type
                generation_method = baseline_type.upper()  # VAE or GAN
            else:
                # For non-baseline models
                model_type = config.get('model', {}).get('type', 'unet')
                # Determine if using Flow Matching or DDPM
                flow_matching_enabled = config.get('flow_matching', {}).get('enabled', False)
                ddpm_enabled = config.get('ddpm', {}).get('enabled', False)
                generation_method = "Flow Matching" if flow_matching_enabled else "DDPM" if ddpm_enabled else "Unknown"

            # Extract data parameters
            data_config = config.get('data', {})
            model_config = config.get('model', {})
            training_config = config.get('training', {})
            ddpm_config = config.get('ddpm', {})
            condition_config = config.get('condition', {})
            fw_config = config.get('flow_matching', {})

            # Extract key configuration information
            region_dict = {
                "Chengdu": "Chengdu",
                "XiAn": "XiAn",
            }
            config_info = {
                'model_type': model_type,
                'generation_method': generation_method,
                'region': region_dict.get(data_config.get('region', 'Unknown'), data_config.get('region', 'Unknown')),
                'fw_sampling_steps': inference_config.get('num_steps', 'unknown'),  # Fixed key
                'ddim_sampling_steps': ddpm_config.get('ddim_steps', 'unknown'),  # Fixed key
                'sample_count': data_config.get('sample_count', 'unknown'),
                'trajectory_length': data_config.get('trajectory_length', 'unknown'),
                'conditional': condition_config.get('enabled', 'unknown'),
                'hidden_dim': model_config.get('hidden_dim', 'unknown'),
                'batch_size': data_config.get('batch_size', 'unknown'),
                'epochs': training_config.get('num_epochs', 'unknown'),
                'learning_rate': training_config.get('learning_rate', 'unknown'),
                'dropout_prob': fw_config.get('dropout_prob', 'unknown'),
                'condition_type': condition_config.get('condition_type', 'unknown'),
                'embedding_type': condition_config.get('embedding_type', 'unknown'),
                'flow_matching_enabled': config.get('flow_matching', {}).get('enabled', False),
                'ddpm_enabled': config.get('ddpm', {}).get('enabled', False),
                'baseline_enabled': baseline_enabled,
                'parametrized': data_config.get('parametrized', 'unknown'),
                'parametrized_method': data_config.get('parametrized_method', 'unknown'),
                'parametrized_M': data_config.get('parametrized_M', 'unknown'),
                'norm1by1': data_config.get('norm1by1', 'unknown'),
                'od_finer': data_config.get('od_finer', 'unknown'),
                'geohash': data_config.get('geohash', 'unknown'),
                'AOITYPE': data_config.get('AOITYPE', 'unknown'),
                'AOIEMB': data_config.get('AOIEMB', 'unknown')
            }
        except Exception as e:
            print(f"Error loading config from {config_file}: {e}")

    return config_info
def evaluate_single_folder(folder_path, args):
    """Evaluate a single generation folder"""
    print(f"Evaluating: {folder_path}")

    # Check required files
    gen_csv = os.path.join(folder_path, "generated_trajectories.csv")
    gt_csv = os.path.join(folder_path, "raw_ground_truth_trajectories.csv")

    if not os.path.exists(gen_csv) or not os.path.exists(gt_csv):
        print(f"✗ Missing files in {folder_path}")
        print(f"  Generated: {os.path.exists(gen_csv)}")
        print(f"  Ground truth: {os.path.exists(gt_csv)}")
        return None

    # Load data
    gt_df = pd.read_csv(gt_csv)
    gen_df = pd.read_csv(gen_csv)

    # Optionally limit trajectory count for evaluation.
    max_trajs = getattr(args, 'max_trajs', 3000)
    if max_trajs > 0:
        gt_uids = gt_df['uid'].unique()[:max_trajs]
        gen_uids = gen_df['uid'].unique()[:max_trajs]
        gt_df = gt_df[gt_df['uid'].isin(gt_uids)]
        gen_df = gen_df[gen_df['uid'].isin(gen_uids)]

    # Load config information
    config_info = load_config_info(folder_path)

    # Set mesh degree for JISMESH
    if args.division_type.upper() == "JISMESH":
        jismesh_switcher = {80000: 1, 40000: 40, 20000: 20, 16000: 16,
                            10000: 2, 8000: 8, 5000: 5, 4000: 4, 2500: 2.5, 2000: 2,
                            1000: 3, 500: 4, 250: 5, 125: 6}
        mesh_degree = jismesh_switcher.get(args.mesh_size, 3)
    else:
        mesh_degree = -1

    VIS_ONLY = False
    if VIS_ONLY:
        # Create visualizations only
        vis_dir = os.path.join(os.path.dirname(folder_path), 'visualizations', os.path.basename(folder_path))
        os.makedirs(vis_dir, exist_ok=True)
        create_comprehensive_visualizations(gt_df, gen_df, vis_dir, args)
        print(f"✓ Visualizations created in: {vis_dir}")
        return None
    # Calculate metrics
    density_js = calculate_density_error(
        gt_df.copy(), gen_df.copy(),
        consider_hour=False,
        division_type=args.division_type,
        mesh_degree=mesh_degree,
        grid_num=args.grid_num
    )

    trip_js = calculate_trip_error(
        gt_df.copy(), gen_df.copy(),
        division_type=args.division_type,
        mesh_degree=mesh_degree,
        grid_num=args.grid_num
    )

    pattern_score = calcu_pattern_score(
        gt_df.copy(), gen_df.copy(),
        top_n=args.top_n,
        division_type=args.division_type,
        mesh_degree=mesh_degree,
        grid_num=args.grid_num
    )

    len_err = length_error(gt_df, gen_df)

    # Calculate new DTW and Fréchet metrics
    print("  Calculating DTW distance...")
    dtw_stats, dtw_vals = calculate_dtw_distance(gt_df.copy(), gen_df.copy(), sample_size=args.DTW_sample_size)

    print("  Calculating Fréchet distance...")
    fr_stats, fr_vals = calculate_frechet_distance(gt_df.copy(), gen_df.copy(), sample_size=args.DTW_sample_size)

    dtw_sum = summarize_indicator(dtw_vals)
    fr_sum = summarize_indicator(fr_vals)
    dtw_median, dtw_mean, dtw_iqr, dtw_std, dtw_p10, dtw_p90 = (
        dtw_sum['median'], dtw_sum['mean'], dtw_sum['iqr'], dtw_sum['std'], dtw_sum['p10'], dtw_sum['p90'])
    fr_median, fr_mean, fr_iqr, fr_std, fr_p10, fr_p90 = (
        fr_sum['median'], fr_sum['mean'], fr_sum['iqr'], fr_sum['std'], fr_sum['p10'], fr_sum['p90'])

    # Create visualizations
    # Create visualization directory
    vis_dir = os.path.join(os.path.dirname(folder_path), 'visualizations', os.path.basename(folder_path))
    os.makedirs(vis_dir, exist_ok=True)
    plot_quantile_curve(dtw_vals, title='DTW quantiles',
                        save_path=os.path.join(vis_dir, 'dtw_quantiles.png'))
    plot_quantile_curve(fr_vals, title='Frechet quantiles',
                        save_path=os.path.join(vis_dir, 'frechet_quantiles.png'))

    # Save individual results with config information
    # get name from folder_path
    folder_name = os.path.basename(folder_path)
    results = {
        'experiment': config_info.get('experiment', os.path.basename(os.path.dirname(folder_path))),
        'eval_name': folder_name,
        'condition': config_info.get('condition', 'unknown'),
        'model_type': config_info.get('model_type', 'unet'),
        'generation_method': config_info.get('generation_method', 'DDPM'),
        'parametrized_method': config_info.get('parametrized_method', 'rdp_k'),
        'region': config_info.get('region', 'Unknown'),
        'length_error': len_err,
        'pattern_score': pattern_score,
        'trip_js': trip_js,
        'density_js': density_js,
        'dtw_mean': dtw_mean,  # New metric
        'frechet_mean': fr_mean,  # New metric
        'dtw_median': dtw_median,
        'frechet_median': fr_median,
        'mesh_degree': mesh_degree,
        'division_type': args.division_type,
        'top_n': args.top_n,
        'grid_num': args.grid_num,
        # dtw and fréchet detailed stats
        'dtw_iqr': dtw_iqr,
        'dtw_std': dtw_std,
        'dtw_p10': dtw_p10,
        'dtw_p90': dtw_p90,
        'fr_iqr': fr_iqr,
        'fr_std': fr_std,
        'fr_p10': fr_p10,
        'fr_p90': fr_p90,
        #'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #'folder_path': folder_path
    }

    results.update(config_info)  # Merge config info into results

    # Save individual evaluation results
    eval_file = os.path.join(folder_path, 'evaluation_results.yaml')
    import yaml
    with open(eval_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"✓ Completed: {folder_path}")
    print(f"  Model: {results['model_type']} | Method: {results['generation_method']} | Region: {results['region']}")
    print(f"  Length Error: {len_err:.6f}")
    print(f"  Pattern Score: {pattern_score:.6f}")
    print(f"  Trip JS: {trip_js:.6f}")
    print(f"  Density JS: {density_js:.6f}")
    print(f"  DTW Distance: {dtw_median:.6f}")
    print(f"  Fréchet Distance: {fr_median:.6f}")

    create_comprehensive_visualizations(gt_df, gen_df, vis_dir, args)


    return results

    # except Exception as e:
    #     print(f"✗ Error evaluating {folder_path}: {e}")
    #     return None


def create_summary_csv(all_results, output_path):
    """Create CSV summary of all evaluation results"""
    if not all_results:
        print("No results to create summary from!")
        return None

    summary_data = []
    for result in all_results:
        if result is not None:
            # Extract all fields including mesh_size_param
            summary_record = {
                'experiment': result['experiment'],
                'eval_name': result['eval_name'],
                'length_error': result['length_error'],
                'pattern_score': result['pattern_score'],
                'trip_js': result['trip_js'],
                'density_js': result['density_js'],
                'division_type': result['division_type'],
                'mesh_size': result['mesh_size_param'],
                # 'grid_num': result['grid_num'],
                # Fixed config columns with correct keys
                'model_type': result.get('model_type', 'unknown'),
                'generation_method': result.get('generation_method', 'unknown'),  # Changed from 'method'
                'region': result.get('region', 'unknown'),
                'fw_step': result.get('fw_sampling_steps', 'unknown'),  # Fixed key
                'ddim_step': result.get('ddim_sampling_steps', 'unknown'),  # Fixed key
                'sample_count': result.get('sample_count', 'unknown'),
                'trajectory_length': result.get('trajectory_length', 'unknown'),
                'conditional': result.get('conditional', False),
                'hidden_dim': result.get('hidden_dim', 'unknown'),
                'batch_size': result.get('batch_size', 'unknown'),
                'epochs': result.get('epochs', 'unknown'),
                'learning_rate': result.get('learning_rate', 'unknown'),
                'dropout_prob': result.get('dropout_prob', 'unknown'),
                'condition_type': result.get('condition_type', 'unknown'),
                'embedding_type': result.get('embedding_type', 'unknown'),
                'flow_matching_enabled': result.get('flow_matching_enabled', False),
                'ddpm_enabled': result.get('ddpm_enabled', False),
                'parametrized': result.get('parametrized', False),
                'parametrized_method': result.get('parametrized_method', 'unknown'),
                'parametrized_M': result.get('parametrized_M', 'unknown'),
                'norm1by1': result.get('norm1by1', False),
                'od_finer': result.get('od_finer', False),
                'geohash': result.get('geohash', False),
                'AOITYPE': result.get('AOITYPE', False),
                'AOIEMB': result.get('AOIEMB', False),
                #'evaluation_timestamp': result.get('evaluation_timestamp', ''),
                #'folder_path': result.get('folder_path', '')
                # get dtw and fréchet metrics
                'dtw_mean': result.get('dtw_mean', 'unknown'),
                'frechet_mean': result.get('frechet_mean', 'unknown'),
                'dtw_median': result.get('dtw_median', 'unknown'),
                'frechet_median': result.get('frechet_median', 'unknown'),
                'dtw_iqr': result.get('dtw_iqr', 'unknown'),
                'dtw_std': result.get('dtw_std', 'unknown'),
                'dtw_p10': result.get('dtw_p10', 'unknown'),
                'dtw_p90': result.get('dtw_p90', 'unknown'),
                'fr_iqr': result.get('fr_iqr', 'unknown'),
                'fr_std': result.get('fr_std', 'unknown'),
                'fr_p10': result.get('fr_p10', 'unknown'),
                'fr_p90': result.get('fr_p90', 'unknown')
            }
            summary_data.append(summary_record)

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_path, index=False)
        print(f"Summary CSV saved to: {output_path}")

        #export a brief version
        # when a columns'values are all the same, we can drop it
        brief_columns = []
        for col in df_summary.columns:
            if df_summary[col].nunique() == 1:
                brief_columns.append(col)
        df_brief = df_summary.drop(columns=brief_columns)
        brief_output_path = output_path.replace('.csv', '_brief.csv')
        df_brief.to_csv(brief_output_path, index=False)

        # Print summary statistics
        print(f"\nSUMMARY STATISTICS ({len(summary_data)} experiments):")
        print("=" * 80)
        for metric in ['length_error', 'pattern_score', 'trip_js', 'density_js']:
            mean_val = df_summary[metric].mean()
            std_val = df_summary[metric].std()
            print(f"{metric.upper()}: mean={mean_val:.6f}, std={std_val:.6f}")

        # Group by different categories
        print("\n" + "=" * 80)
        print("BREAKDOWN BY MODEL TYPE:")
        print("=" * 80)
        model_groups = df_summary.groupby('model_type')
        for model, group in model_groups:
            print(f"\n{model.upper()} ({len(group)} experiments):")
            for metric in ['length_error', 'pattern_score', 'trip_js', 'density_js']:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                print(f"  {metric}: mean={mean_val:.6f}, std={std_val:.6f}")

        print("\n" + "=" * 80)
        print("BREAKDOWN BY GENERATION METHOD:")
        print("=" * 80)
        method_groups = df_summary.groupby('generation_method')
        for method, group in method_groups:
            print(f"\n{method.upper()} ({len(group)} experiments):")
            for metric in ['length_error', 'pattern_score', 'trip_js', 'density_js']:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                print(f"  {metric}: mean={mean_val:.6f}, std={std_val:.6f}")

        print("\n" + "=" * 80)
        print("BREAKDOWN BY PARAMETRIZED METHOD:")
        print("=" * 80)
        param_groups = df_summary.groupby('parametrized_method')
        for param_method, group in param_groups:
            print(f"\n{param_method} ({len(group)} experiments):")
            for metric in ['length_error', 'pattern_score', 'trip_js', 'density_js']:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                print(f"  {metric}: mean={mean_val:.6f}, std={std_val:.6f}")

        print("\n" + "=" * 80)
        print("BREAKDOWN BY REGION:")
        print("=" * 80)
        region_groups = df_summary.groupby('region')
        for region, group in region_groups:
            print(f"\n{region} ({len(group)} experiments):")
            for metric in ['length_error', 'pattern_score', 'trip_js', 'density_js']:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                print(f"  {metric}: mean={mean_val:.6f}, std={std_val:.6f}")

        print("=" * 80)

        return df_summary

    return None
def find_generation_folders(base_dir, mode, exp_list=None, generate_num=5000, condition_mode='real'):
    """Find generation folders based on mode and criteria"""
    folders = []

    if mode == '1':
        # Single experiment - exp_list[0] is the experiment name
        if exp_list and len(exp_list) > 0:
            exp_name = exp_list[0]
            exp_dir = os.path.join(base_dir, exp_name)
            if os.path.exists(exp_dir):
                for per_condition_mode in condition_mode:
                    folder_pattern = f"generation_{generate_num}_{per_condition_mode}"
                    for folder_name in os.listdir(exp_dir):
                        if folder_pattern in folder_name:
                            gen_folder = os.path.join(exp_dir, folder_name)
                            if os.path.exists(gen_folder) and gen_folder not in folders:
                                folders.append(gen_folder)
    elif mode == '2':
        # Multiple specific experiments
        if exp_list:
            for exp_name in exp_list:
                exp_dir = os.path.join(base_dir, exp_name)
                if os.path.exists(exp_dir):
                    for per_condition_mode in condition_mode:
                        folder_pattern = f"generation_{generate_num}_{per_condition_mode}"
                        for folder_name in os.listdir(exp_dir):
                            if folder_pattern in folder_name:
                                gen_folder = os.path.join(exp_dir, folder_name)
                                if os.path.exists(gen_folder) and gen_folder not in folders:
                                    folders.append(gen_folder)

    elif mode == '3':
        # All experiments
        import glob
        for exp_dir in glob.glob(os.path.join(base_dir, '*')):
            if os.path.isdir(exp_dir):
                for per_condition_mode in condition_mode:
                    folder_pattern = f"generation_{generate_num}_{per_condition_mode}"
                    for folder_name in os.listdir(exp_dir):
                        if folder_pattern in folder_name:
                            gen_folder = os.path.join(exp_dir, folder_name)
                            if os.path.exists(gen_folder) and gen_folder not in folders:
                                folders.append(gen_folder)

    return folders


def main():
    """Main function with mode support"""
    parser = argparse.ArgumentParser(description='Evaluate generated trajectory results')

    # Mode selection
    parser.add_argument('--mode', type=str, default='1', choices=['1', '2', '3'],
                        help='Evaluation mode: 1=single exp, 2=multiple exp, 3=all exp')
    parser.add_argument('--exp_list', type=str, nargs='+',
                        default=['run_20250916_133645'], #, 'run_20250801_213941', 'run_20250801_213957',,'04-23-14-32-05','04-25-12-44-58'
                        help='Experiment names (required for mode 1&2)')
    parser.add_argument('--generate_num', type=int, default=5000,
                        help='Generation number used in folder naming')
    parser.add_argument('--condition_mode', type=str, nargs='+',
                        default=['real'], #, 'placeholder', 'CAR', 'TRAIN'
                        help='Condition mode used in generation')
    parser.add_argument('--base_dir', type=str,
                        default='./generate_results_baseline_full_clean/',
                        help='Base directory containing generated results')

    # Legacy parameters for backward compatibility
    parser.add_argument('--time_str', type=str, default=None,
                        help='Time string (legacy, use --exp_list instead)')

    # Evaluation parameters
    parser.add_argument('--division_type', type=str, default='JISMESH',
                        choices=['DIVISION', 'JISMESH'],
                        help='Grid division method for evaluation')
    parser.add_argument('--mesh_size', type=int, nargs='+', default=[1000],#
                        help='Mesh size(s) for JISMESH division - can specify multiple values')
    parser.add_argument('--grid_num', type=int, default=16,
                        help='Number of grid divisions for DIVISION method')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Top N regions for pattern score calculation')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv',
                        help='Output CSV file for evaluation results')
    parser.add_argument('--DTW_sample_size', type=int, default=2000,
                        help='Sample size for DTW and Fréchet distance calculations')
    args = parser.parse_args()

    # Handle legacy time_str parameter
    if args.time_str and not args.exp_list:
        args.exp_list = [args.time_str]
        if not args.mode or args.mode == '1':
            args.mode = '1'

    # Validate arguments
    if args.mode in ['1', '2'] and not args.exp_list:
        print(f"Error: --exp_list is required for mode {args.mode}")
        return

    # Find generation folders
    folders = find_generation_folders(
        args.base_dir, args.mode, args.exp_list,
        args.generate_num, args.condition_mode
    )

    if not folders:
        print("No generation folders found!")
        print(f"Looking in: {args.base_dir}")
        if args.exp_list:
            print(f"For experiments: {args.exp_list}")
        print(f"Pattern: generation_{args.generate_num}_{args.condition_mode}")
        return

    print(f"Found {len(folders)} folders to evaluate")
    for folder in folders:
        print(f"  - {folder}")

    # Run evaluations for each mesh size
    all_results = []

    # Convert mesh_size to list if it's a single value
    mesh_sizes = args.mesh_size if isinstance(args.mesh_size, list) else [args.mesh_size]

    print(f"\nTesting mesh sizes: {mesh_sizes}")

    for mesh_size in mesh_sizes:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING WITH MESH SIZE: {mesh_size}")
        print(f"{'=' * 60}")

        # Update args for current mesh size
        args.mesh_size = mesh_size

        for folder in folders:
            result = evaluate_single_folder(folder, args)
            if result:
                result['mesh_size_param'] = mesh_size  # Track which mesh size was used
                all_results.append(result)

    # Create summary for multiple experiments
    if len(folders) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_dir = os.path.join(args.base_dir, f"evaluation_summary_{timestamp}")
        os.makedirs(summary_dir, exist_ok=True)

        print(f"\nCreating summary in: {summary_dir}")
        summary_file = os.path.join(summary_dir, 'mesh_size_comparison_summary.csv')
        df_summary = create_summary_csv(all_results, summary_file)

        # Create mesh size comparison analysis
        try:
            if df_summary is not None:
                create_mesh_size_analysis(df_summary, summary_dir, mesh_sizes)
        except:
            print("Failed to create mesh size analysis visualizations.")

    print(
        f"\nEvaluation completed! Processed {len([r for r in all_results if r is not None])} evaluations successfully.")
    print(
        f"Total combinations: {len(folders)} folders × {len(mesh_sizes)} mesh sizes = {len(folders) * len(mesh_sizes)}")

if __name__ == "__main__":
    main()
