"""Module for optimization metric calculations"""
from typing import List, Tuple
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

def calculate_spatial_equity(points: List[Point]) -> float:
    """Calculate spatial equity metric"""
    if len(points) < 2:
        return 0.0
        
    distances = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points[i+1:], i+1):
            distances.append(p1.distance(p2))
            
    return np.std(distances)

def calculate_coverage(points: List[Point], 
                     population_raster: str,
                     service_radius: float) -> float:
    """Calculate population coverage metric"""
    with rasterio.open(population_raster) as src:
        population = src.read(1)
        transform = src.transform
        
        # Create buffers around points
        gdf = gpd.GeoDataFrame(geometry=[
            point.buffer(service_radius) for point in points
        ])
        
        # Rasterize buffers
        shapes = [(geom, 1) for geom in gdf.geometry]
        coverage = rasterize(
            shapes,
            out_shape=population.shape,
            transform=transform,
            dtype=np.uint8
        )
        
        # Calculate covered population
        total_pop = np.sum(population)
        covered_pop = np.sum(population * coverage)
        
        return (covered_pop / total_pop) if total_pop > 0 else 0.0