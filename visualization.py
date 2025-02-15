"""Module for visualization functions"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import geopandas as gpd
from typing import List, Dict
import numpy as np
from pathlib import Path

def plot_optimization_results(metrics: Dict,
                            output_dir: Path,
                            font: FontProperties) -> None:
    """Create visualization plots for optimization results"""
    # Pareto front visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        metrics['equity'],
        metrics['coverage'],
        metrics['cost'],
        c=metrics['cost'],
        cmap='viridis'
    )
    
    ax.set_xlabel('空间均衡性', fontproperties=font)
    ax.set_ylabel('服务人口覆盖', fontproperties=font)
    ax.set_zlabel('建设成本', fontproperties=font)
    
    plt.colorbar(scatter)
    plt.title('优化结果三维可视化', fontproperties=font)
    plt.savefig(output_dir / 'optimization_results.png', dpi=300)
    plt.close()