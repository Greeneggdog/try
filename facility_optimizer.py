import os
import numpy as np
from datetime import datetime
import random
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from deap import base, creator, tools, algorithms
import pandas as pd
import chardet
import pyproj
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import accessibility_analysis
from shapely.geometry import Point

@dataclass
class OptimizationParameters:
    """Data class for optimization parameters"""
    existing_facilities: str
    population_raster: str
    accessibility_raster: str
    land_cost_raster: str
    new_facility_count: int
    service_radius: float
    output_folder: str
    population_size: int = 50
    generations: int = 20
    underserved_threshold_percentile: int = 60
    num_candidates: int = 200
    shanghai_boundary: str = ""

class FacilityOptimizer:
    """Medical facility location optimizer using NSGA-II algorithm"""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the medical facility optimizer"""
        self.logger = self._setup_logger(log_level)
        self.label = '医疗设施多目标选址优化'
        self.description = "使用NSGA-II算法进行医疗设施选址优化，考虑空间均衡性、服务覆盖率和建设成本三个目标"
        self._setup_fonts()
        self.population_raster_path: Optional[str] = None
        self.baseline_metrics: Dict = {}
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:  # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
        return logger

    def _setup_fonts(self) -> None:
        """Configure fonts for visualization with fallback options"""
        try:
            possible_font_paths = [
                Path("D:/Desktop/Study_Material/Thesis_Associated_Resources/fonts/simkai.ttf"),
                Path("C:/Windows/Fonts/simkai.ttf"),
                Path("C:/Windows/Fonts/SimHei.ttf"),
            ]
            
            font_found = False
            for font_path in possible_font_paths:
                if font_path.exists():
                    self.font = FontProperties(fname=str(font_path))
                    font_found = True
                    self.logger.info(f"Using font from: {font_path}")
                    break
            
            if not font_found:
                self.font = FontProperties(family=['SimHei', 'SimSun', 'Microsoft YaHei'])
                self.logger.info("Using system font families")
            
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception as e:
            self.logger.warning(f"Font setup failed: {e}. Using system defaults.")
            self.font = FontProperties()

    def _create_output_folder(self, base_path: Union[str, Path]) -> Optional[Path]:
        """Create output directory with timestamp"""
        try:
            base_path = Path(base_path)
            base_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_path / f"FacilityOptimization_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (output_dir / "Pareto_Solutions").mkdir(exist_ok=True)
            (output_dir / "Visualization").mkdir(exist_ok=True)
            (output_dir / "Comparison").mkdir(exist_ok=True)
            
            self.logger.info(f"Created output directory: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {str(e)}")
            return None

    @staticmethod
    def read_shapefile_safe(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """Safely read a shapefile with automatic encoding detection"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                detected_encoding = chardet.detect(raw_data)['encoding']
            return gpd.read_file(file_path, encoding=detected_encoding)
        except Exception as e:
            try:
                return gpd.read_file(file_path, encoding='GBK')
            except Exception as nested_e:
                raise ValueError(f"Could not read file {file_path}: {str(nested_e)}") from e

    def _identify_underserved_areas(self, accessibility_path: str,
                              population_path: str,
                              underserved_threshold_percentile: int) -> Tuple[np.ndarray, rasterio.Affine]:
        """Identify underserved areas based on accessibility and population"""
        try:
            with rasterio.open(accessibility_path) as src:
                accessibility = src.read(1)
                transform = src.transform
                accessibility_nodata = src.nodata

            with rasterio.open(population_path) as src:
                population = src.read(1)
                population_nodata = src.nodata

            # 修正：更改NoData值处理方式
            if accessibility_nodata is not None:
                accessibility = np.where(accessibility == accessibility_nodata, 0, accessibility)
            if population_nodata is not None:
                population = np.where(population == population_nodata, 0, population)

            # 修正：只考虑有人口的区域
            valid_mask = (population > 0)
            valid_acc = accessibility[valid_mask]
            
            if len(valid_acc) == 0:
                raise ValueError("No valid areas found with population")

            # 修正：计算分位数时只考虑有效区域
            threshold = np.percentile(valid_acc[valid_acc > 0], underserved_threshold_percentile)
            
            # 修正：更新服务不足区域的识别标准
            underserved = (accessibility <= threshold) & (population > 0)

            # 记录识别到的服务不足区域信息
            total_population = np.sum(population[underserved])
            area_count = np.sum(underserved)
            
            self.logger.info(f"Identified {area_count} underserved cells")
            self.logger.info(f"Total population in underserved areas: {total_population}")

            return underserved, transform

        except Exception as e:
            self.logger.error(f"Error in identifying underserved areas: {str(e)}")
            raise
    def _generate_candidate_subset(self, rows: np.ndarray, cols: np.ndarray,
                             transform: rasterio.Affine,
                             boundary_path: str) -> List[Point]:
        """Generate a subset of candidate points with improved distribution"""
        try:
            boundary = self.read_shapefile_safe(boundary_path)
            
            with rasterio.open(self.population_raster_path) as src:
                raster_crs = src.crs
                population = src.read(1)
                
            if boundary.crs != raster_crs:
                boundary = boundary.to_crs(raster_crs)
                
            boundary_polygon = boundary.geometry.union_all()
            
            candidates = []
            weights = []  # 添加权重来优化候选点分布
            
            for row, col in zip(rows, cols):
                x, y = transform * (col, row)
                point = Point(x, y)
                
                if point.within(boundary_polygon):
                    # 获取该点位置的人口数作为权重
                    pop_value = population[row, col]
                    candidates.append(point)
                    weights.append(pop_value if pop_value > 0 else 1)
                    
            # 如果有足够的候选点，使用权重进行随机选择
            if len(candidates) > 0:
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # 归一化权重
                
                # 记录候选点的分布情况
                self.logger.info(f"Generated {len(candidates)} candidate points")
                if len(candidates) > 0:
                    bbox = boundary_polygon.bounds
                    x_range = bbox[2] - bbox[0]
                    y_range = bbox[3] - bbox[1]
                    
                    x_coords = [p.x for p in candidates]
                    y_coords = [p.y for p in candidates]
                    
                    x_std = np.std(x_coords) / x_range
                    y_std = np.std(y_coords) / y_range
                    
                    self.logger.info(f"Candidate points distribution - X std: {x_std:.3f}, Y std: {y_std:.3f}")
            
            return candidates

        except Exception as e:
            self.logger.error(f"Error generating candidate subset: {str(e)}")
            return []

    def _generate_candidate_subset(self, rows: np.ndarray, cols: np.ndarray,
                             transform: rasterio.Affine,
                             boundary_path: str) -> List[Point]:
        """Generate a subset of candidate points"""
        try:
            # Read boundary file with proper CRS
            boundary = self.read_shapefile_safe(boundary_path)
            
            with rasterio.open(self.population_raster_path) as src:
                raster_crs = src.crs
                
            # Ensure boundary is in the same CRS as the raster
            if boundary.crs != raster_crs:
                boundary = boundary.to_crs(raster_crs)
                
            # Convert to UTM for accurate spatial operations if in geographic coordinates
            if boundary.crs.is_geographic:
                # Calculate UTM zone from centroid after converting to geographic coordinates
                center_lon = boundary.to_crs(4326).geometry.centroid.x.mean()
                utm_zone = int((center_lon + 180) / 6) + 1
                utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
                boundary = boundary.to_crs(utm_crs)
                
            boundary_polygon = boundary.geometry.union_all()
            
            candidates = []
            for row, col in zip(rows, cols):
                x, y = transform * (col, row)
                point = Point(x, y)
                # Create a GeoDataFrame for the point to handle CRS properly
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs=raster_crs)
                
                # Convert point to same CRS as boundary if needed
                if point_gdf.crs != boundary.crs:
                    point_gdf = point_gdf.to_crs(boundary.crs)
                    
                if point_gdf.geometry.iloc[0].within(boundary_polygon):
                    candidates.append(point)
                    
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error generating candidate subset: {str(e)}")
            return []

    def _generate_candidates_parallel(self, underserved_area: np.ndarray,
                                   transform: rasterio.Affine,
                                   radius: float,
                                   num_candidates: int,
                                   boundary_path: str,
                                   max_workers: int = 4) -> List[Point]:
        """Generate candidate points in parallel"""
        rows, cols = np.where(underserved_area)
        points_per_worker = num_candidates // max_workers
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(max_workers):
                start_idx = i * points_per_worker
                end_idx = start_idx + points_per_worker
                futures.append(
                    executor.submit(
                        self._generate_candidate_subset,
                        rows[start_idx:end_idx],
                        cols[start_idx:end_idx],
                        transform,
                        boundary_path
                    )
                )
                
        candidates = []
        for future in futures:
            candidates.extend(future.result())
            
        return candidates[:num_candidates]

    def _calculate_distances(self, points: List[Point]) -> List[float]:
        """Calculate distances between points with improved method"""
        try:
            if len(points) < 2:
                return []
                
            distances = []
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points[i+1:], i+1):
                    distances.append(p1.distance(p2))
                    
            # 添加到边界的距离考虑
            # min_boundary_dist = min(distances) if distances else 0
            return distances
            
        except Exception as e:
            self.logger.error(f"Error calculating distances: {str(e)}")
            return []

    def _calculate_coverage(self, points: List[Point],
                          population_raster: str,
                          service_radius: float,
                          transform: Optional[rasterio.Affine] = None) -> float:
        """Calculate population coverage"""
        with rasterio.open(population_raster) as src:
            population = src.read(1)
            if transform is None:
                transform = src.transform
            
            gdf = gpd.GeoDataFrame(geometry=points)
            gdf.set_crs(src.crs, inplace=True)
            
            if gdf.crs.is_geographic:
                utm_zone = int((points[0].x + 180) / 6) + 1
                utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
                gdf = gdf.to_crs(utm_crs)
            
            buffers = gdf.geometry.buffer(service_radius)
            if gdf.crs != src.crs:
                buffers = gpd.GeoSeries(buffers, crs=gdf.crs).to_crs(src.crs)
            
            shapes = [(geom, 1) for geom in buffers]
            covered = rasterize(shapes, out_shape=population.shape,
                              transform=transform, all_touched=True)
            
            return np.sum(population * covered)

    def _calculate_cost(self, points: List[Point],
                       cost_raster: str,
                       transform: rasterio.Affine) -> float:
        """Calculate total cost"""
        with rasterio.open(cost_raster) as src:
            cost_data = src.read(1)
            total_cost = 0
            
            for point in points:
                row, col = rasterio.transform.rowcol(transform, point.x, point.y)
                if 0 <= row < cost_data.shape[0] and 0 <= col < cost_data.shape[1]:
                    total_cost += cost_data[row, col]
                    
            return total_cost
    def _evaluate_solution(self, individual: List[int],
                     candidates: List[Point],
                     params: OptimizationParameters,
                     transform: rasterio.Affine) -> Tuple[float, float, float]:
        """Evaluate a solution's fitness with improved spatial distribution consideration"""
        try:
            selected_points = [candidates[i] for i in individual]
            
            if len(selected_points) < 2:
                return float('inf'), 0, float('inf')
            
            # 1. 空间均衡性评估改进
            distances = self._calculate_distances(selected_points)
            if not distances:
                return float('inf'), 0, float('inf')
                
            equity = np.std(distances)
            
            # 添加空间分布惩罚项
            bbox = self._get_boundary_bbox(params.shanghai_boundary)
            x_coords = [p.x for p in selected_points]
            y_coords = [p.y for p in selected_points]
            
            x_range = bbox[2] - bbox[0]
            y_range = bbox[3] - bbox[1]
            
            # 计算分布的集中度
            x_std = np.std(x_coords) / x_range
            y_std = np.std(y_coords) / y_range
            
            # 如果分布过于集中，增加惩罚
            distribution_penalty = 1.0
            if x_std < 0.1 or y_std < 0.1:  # 分布过于集中
                distribution_penalty = 2.0
            
            equity *= distribution_penalty
            
            # 2. 覆盖率评估
            coverage = self._calculate_coverage(selected_points,
                                            params.population_raster,
                                            params.service_radius,
                                            transform)
            
            # 3. 成本评估
            cost = self._calculate_cost(selected_points,
                                    params.land_cost_raster,
                                    transform)
            
            return equity, coverage, cost
            
        except Exception as e:
            self.logger.error(f"Error in solution evaluation: {str(e)}")
            return float('inf'), 0, float('inf')

    def _run_optimization(self, candidates: List[Point],
                        params: OptimizationParameters,
                        transform: rasterio.Affine) -> Tuple[List, tools.Logbook]:
        """Run the NSGA-II optimization"""
        try:
            # Clear any existing DEAP creators
            if 'FitnessMulti' in creator.__dict__:
                del creator.FitnessMulti
            if 'Individual' in creator.__dict__:
                del creator.Individual
                
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)
            
            toolbox = base.Toolbox()
            
            def create_individual():
                return creator.Individual(random.sample(range(len(candidates)), params.new_facility_count))
            
            toolbox.register("individual", create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", self._evaluate_solution,
                            candidates=candidates, params=params, transform=transform)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
            toolbox.register("select", tools.selNSGA2)
            
            pop = toolbox.population(n=params.population_size)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            
            with tqdm(total=params.generations) as pbar:
                pop, logbook = algorithms.eaMuPlusLambda(
                    pop, toolbox,
                    mu=params.population_size,
                    lambda_=params.population_size * 2,
                    cxpb=0.7, mutpb=0.3,
                    ngen=params.generations,
                    stats=stats,
                    verbose=True
                )
                pbar.update(params.generations)
            
            pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if not pareto_front:
                raise ValueError("No valid solutions found in Pareto front")
                
            return pareto_front, logbook
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return [], tools.Logbook()

    def _get_boundary_bbox(self, boundary_path: str) -> Tuple[float, float, float, float]:
        """Get the bounding box of the study area"""
        try:
            boundary = self.read_shapefile_safe(boundary_path)
            return boundary.total_bounds
        except Exception as e:
            self.logger.error(f"Error getting boundary bbox: {str(e)}")
            return (0, 0, 1, 1)
    
    

    def _calculate_baseline_metrics(self, existing_facilities_path: str,
                                accessibility_raster_path: str,
                                population_raster_path: str,
                                service_radius: float,
                                shanghai_boundary_path: str) -> None:
        """Calculate baseline metrics before optimization"""
        try:
            # 1. Accessibility metrics
            analyzer = accessibility_analysis.AccessibilityAnalyzer()
            equity_stats = analyzer.calculate_equity_metrics(
                accessibility_raster_path, 
                population_raster_path
            )
            if equity_stats:
                self.baseline_metrics.update(equity_stats)

            # 2. Service coverage
            with rasterio.open(population_raster_path) as src:
                population = src.read(1)
                pop_transform = src.transform
                pop_crs = src.crs

                existing_facilities = self.read_shapefile_safe(existing_facilities_path)
                if existing_facilities.crs != pop_crs:
                    existing_facilities = existing_facilities.to_crs(pop_crs)

                shanghai_boundary = self.read_shapefile_safe(shanghai_boundary_path)
                if shanghai_boundary.crs != pop_crs:
                    shanghai_boundary = shanghai_boundary.to_crs(pop_crs)
                shanghai_polygon = shanghai_boundary.geometry.union_all()

                if existing_facilities.crs.is_geographic:
                    utm_zone = int((existing_facilities.geometry.centroid.x.mean() + 180) / 6) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
                    existing_facilities = existing_facilities.to_crs(utm_crs)
                    shanghai_polygon = gpd.GeoSeries(shanghai_polygon, crs=existing_facilities.crs).to_crs(utm_crs)[0]

                buffers = existing_facilities.geometry.buffer(service_radius)

                if pop_crs.is_geographic:
                    buffers = gpd.GeoSeries(buffers, crs=existing_facilities.crs).to_crs(pop_crs)
                    existing_facilities = existing_facilities.to_crs(pop_crs)
                    shanghai_polygon = gpd.GeoSeries(shanghai_polygon, crs=existing_facilities.crs).to_crs(pop_crs)[0]

                gdf = gpd.GeoDataFrame({'geometry': buffers}, crs=pop_crs)
                gdf = gpd.clip(gdf, shanghai_polygon)

                shapes = [(geom, 1) for geom in gdf.geometry]
                covered_raster = rasterize(shapes, out_shape=population.shape,
                                         transform=pop_transform, all_touched=True,
                                         dtype=np.uint8)

                total_population = np.sum(population)
                covered_population = np.sum(population * covered_raster)
                coverage_ratio = (covered_population / total_population) * 100 if total_population > 0 else 0
                self.baseline_metrics['coverage_ratio'] = coverage_ratio

            # 3. Spatial equity
            if not existing_facilities.empty:
                if existing_facilities.crs.is_geographic:
                    utm_zone = int((existing_facilities.geometry.centroid.x.mean() + 180) / 6) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
                    existing_facilities_proj = existing_facilities.to_crs(utm_crs)
                else:
                    existing_facilities_proj = existing_facilities

                nearest_distances = []
                for point in existing_facilities_proj.geometry:
                    distances = existing_facilities_proj.geometry.distance(point)
                    distances = distances[distances > 0]
                    if not distances.empty:
                        nearest_distances.append(distances.min())

                avg_nearest_neighbor_distance = np.mean(nearest_distances) if nearest_distances else 0
            else:
                avg_nearest_neighbor_distance = 0

            self.baseline_metrics['avg_nearest_neighbor_distance'] = avg_nearest_neighbor_distance
            self.logger.info(f"Baseline metrics calculated: {self.baseline_metrics}")

        except Exception as e:
            self.logger.error(f"Error calculating baseline metrics: {str(e)}")
            self.baseline_metrics = {}

    def _compare_solutions(self, pareto_front: List,
                         candidates: List[Point],
                         comparison_dir: Path,
                         params: OptimizationParameters) -> None:
        """Compare optimization results and generate comparison metrics"""
        try:
            if not pareto_front:
                self.logger.warning("No solutions in Pareto front to compare")
                return

            # Generate comparison metrics
            best_solution = pareto_front[0]  # Use first solution for comparison
            if not best_solution:
                self.logger.warning("Best solution is empty")
                return

            best_points = [candidates[i] for i in best_solution]
            if not best_points:
                self.logger.warning("No points in best solution")
                return

            # Calculate metrics
            comparison_metrics = {
                "spatial_equity": {
                    "before": self.baseline_metrics.get('avg_nearest_neighbor_distance', 0),
                    "after": np.std(self._calculate_distances(best_points)) if len(best_points) > 1 else 0
                },
                "coverage": {
                    "before": self.baseline_metrics.get('coverage_ratio', 0),
                    "after": 0  # Initialize with 0
                }
            }

            # Calculate coverage separately to handle potential errors
            try:
                with rasterio.open(params.population_raster) as src:
                    transform = src.transform
                coverage = self._calculate_coverage(
                    best_points,
                    params.population_raster,
                    params.service_radius,
                    transform
                )
                comparison_metrics["coverage"]["after"] = coverage
            except Exception as e:
                self.logger.error(f"Error calculating coverage: {str(e)}")

            # Save comparison metrics
            metrics_df = pd.DataFrame(comparison_metrics)
            metrics_df.to_csv(comparison_dir / "comparison_metrics.csv")

            # Generate comparison visualizations
            self._generate_comparison_plots(comparison_metrics, comparison_dir)

        except Exception as e:
            self.logger.error(f"Error comparing solutions: {str(e)}")
            # Create empty comparison metrics file
            pd.DataFrame({
                "spatial_equity": {"before": 0, "after": 0},
                "coverage": {"before": 0, "after": 0}
            }).to_csv(comparison_dir / "comparison_metrics.csv")

    def _generate_comparison_plots(self, metrics: Dict, output_dir: Path) -> None:
        """Generate comparison visualization plots"""
        try:
            # Bar chart comparing before/after metrics
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Spatial equity comparison
            equity_data = [metrics['spatial_equity']['before'], metrics['spatial_equity']['after']]
            ax1.bar(['优化前', '优化后'], equity_data)
            ax1.set_title('空间均衡性对比', fontproperties=self.font)
            ax1.set_ylabel('平均最近邻距离', fontproperties=self.font)

            # Coverage comparison
            coverage_data = [metrics['coverage']['before'], metrics['coverage']['after']]
            ax2.bar(['优化前', '优化后'], coverage_data)
            ax2.set_title('服务覆盖率对比', fontproperties=self.font)
            ax2.set_ylabel('覆盖率 (%)', fontproperties=self.font)

            plt.tight_layout()
            plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {str(e)}")

    def execute(self, parameters: Dict) -> str:
        """Execute the optimization process"""
        try:
            input_params = OptimizationParameters(**parameters)
            self.population_raster_path = input_params.population_raster

            # Validate input files exist
            required_files = [
                (input_params.existing_facilities, "Existing facilities file"),
                (input_params.population_raster, "Population raster"),
                (input_params.accessibility_raster, "Accessibility raster"),
                (input_params.land_cost_raster, "Land cost raster"),
                (input_params.shanghai_boundary, "Shanghai boundary file")
            ]

            for file_path, file_desc in required_files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"{file_desc} not found: {file_path}")

            output_dir = self._create_output_folder(input_params.output_folder)
            if not output_dir:
                raise ValueError("Failed to create output directory")

            self.logger.info("Starting optimization process...")

            # Calculate baseline metrics
            self._calculate_baseline_metrics(
                input_params.existing_facilities,
                input_params.accessibility_raster,
                input_params.population_raster,
                input_params.service_radius,
                input_params.shanghai_boundary
            )

            # Identify underserved areas
            underserved_area, transform = self._identify_underserved_areas(
                input_params.accessibility_raster,
                input_params.population_raster,
                input_params.underserved_threshold_percentile
            )

            # Generate candidates
            candidates = self._generate_candidates_parallel(
                underserved_area,
                transform,
                input_params.service_radius,
                input_params.num_candidates,
                input_params.shanghai_boundary
            )

            if not candidates:
                raise ValueError("No valid candidate points generated")

            # Run optimization
            pareto_front, logbook = self._run_optimization(
                candidates=candidates,
                params=input_params,
                transform=transform
            )

            # Save results
            self._save_optimization_results(
                pareto_front=pareto_front,
                candidates=candidates,
                logbook=logbook,
                output_dir=output_dir,
                transform=transform,
                params=input_params
            )

            self.logger.info(f"Optimization completed. Results saved to: {output_dir}")
            return str(output_dir)

        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            emergency_dir = Path.cwd() / "Emergency_Output"
            emergency_dir.mkdir(exist_ok=True)
            self.logger.info(f"Emergency results saved to: {emergency_dir}")
            return str(emergency_dir)

    def _save_optimization_results(self, pareto_front: List,
                            candidates: List[Point],
                            logbook: tools.Logbook,
                            output_dir: Path,
                            transform: rasterio.Affine,
                            params: OptimizationParameters) -> None:
        """Save optimization results and generate visualizations"""
        try:
            # Create result directories
            solution_dir = output_dir / "Pareto_Solutions"
            viz_dir = output_dir / "Visualization"
            comparison_dir = output_dir / "Comparison"

            # Get CRS from population raster
            with rasterio.open(params.population_raster) as src:
                crs = src.crs

            # 准备解决方案数据用于HTML报告
            solutions = []
            for idx, ind in enumerate(pareto_front):
                points = [candidates[i] for i in ind]
                gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
                
                if gdf.crs.is_geographic:
                    utm_zone = int((gdf.geometry.centroid.x.mean() + 180) / 6) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                
                filename = f"Solution_{idx + 1}_Equity{ind.fitness.values[0]:.1f}.shp"
                gdf.to_file(solution_dir / filename, encoding='utf-8')
                
                solutions.append({
                    "id": idx + 1,
                    "equity": f"{ind.fitness.values[0]:.2f}",
                    "coverage": f"{ind.fitness.values[1]:.0f}",
                    "cost": f"¥{ind.fitness.values[2]:,.2f}"
                })

            # Generate visualizations
            self._generate_visualizations(pareto_front, logbook, viz_dir)
            
            # Generate comparison results
            comparison_results = self._compare_solutions(pareto_front, candidates, comparison_dir, params)
            
            # Generate HTML report
            self._generate_html_report(solutions, output_dir, comparison_results)

        except Exception as e:
            self.logger.error(f"Error saving optimization results: {str(e)}")

    def _generate_visualizations(self, pareto_front: List,
                               logbook: tools.Logbook,
                               viz_dir: Path) -> None:
        """Generate visualization plots"""
        try:
            # 3D Pareto front visualization
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            equity = [ind.fitness.values[0] for ind in pareto_front]
            coverage = [ind.fitness.values[1] for ind in pareto_front]
            cost = [ind.fitness.values[2] for ind in pareto_front]

            scatter = ax.scatter(equity, coverage, cost, c=cost, cmap='viridis', s=50)

            ax.set_xlabel('空间均衡性（标准差）', fontproperties=self.font, fontsize=10)
            ax.set_ylabel('服务人口覆盖', fontproperties=self.font, fontsize=10)
            ax.set_zlabel('建设成本（元）', fontproperties=self.font, fontsize=10)

            cbar = plt.colorbar(scatter)
            cbar.ax.set_ylabel('建设成本', fontproperties=self.font, fontsize=12)
            plt.title("Pareto前沿三维可视化", fontproperties=self.font, fontsize=14)
            plt.savefig(viz_dir / "3d_pareto_front.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Convergence trend
            plt.figure(figsize=(10, 6))
            generations = logbook.select("gen")
            avg_equity = [entry['avg'][0] for entry in logbook]

            plt.plot(generations, avg_equity, 'b-o', linewidth=2)
            plt.xlabel('迭代次数', fontproperties=self.font, fontsize=12)
            plt.ylabel('平均空间均衡性', fontproperties=self.font, fontsize=12)
            plt.title('优化过程收敛趋势', fontproperties=self.font, fontsize=14)
            plt.grid(True)
            plt.savefig(viz_dir / "convergence_trend.png", dpi=300)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
    def _generate_html_report(self, solutions: List[Dict], output_dir: Path, comparison_results: Dict = None) -> None:
        """生成HTML格式的优化报告，包括优化前后对比"""
        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>医疗设施优化报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            .solution-table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 30px;
            }}
            .solution-table th, .solution-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .solution-table th {{
                background-color: #f8f9fa;
            }}
            .image-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-top: 20px;
            }}
            .image-box {{
                flex: 1 1 45%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
                min-width: 300px;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            .comparison-table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }}
            .comparison-table th, .comparison-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .comparison-table th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>医疗设施选址优化报告</h1>
        <p>生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <h2>Pareto最优解集</h2>
        <table class="solution-table">
            <tr>
                <th>方案ID</th>
                <th>空间均衡性</th>
                <th>覆盖人口</th>
                <th>建设成本</th>
            </tr>
            {"".join(f'''
            <tr>
                <td>{sol['id']}</td>
                <td>{sol['equity']}</td>
                <td>{sol['coverage']}</td>
                <td>{sol['cost']}</td>
            </tr>
            ''' for sol in solutions)}
        </table>
        """

        # 添加对比分析结果
        if comparison_results:
            html_content += """
        <h2>优化前后对比</h2>
        <table class="comparison-table">
            <tr>
                <th>指标</th>
                <th>优化前</th>
                <th>优化后 (最佳方案)</th>
                <th>变化</th>
            </tr>
    """
            for metric, values in comparison_results["metrics"].items():
                before_val = values['before']
                after_val = values['after']
                change_val = values['change']

                before_str = f"{before_val:.2f}" if before_val is not None else "N/A"
                after_str = f"{after_val:.2f}" if after_val is not None else "N/A"
                change_str = f"{change_val:.2f}" if change_val is not None else "N/A"

                if before_val is not None and before_val != 0:
                    percent_change = (after_val - before_val) / before_val * 100 \
                        if after_val is not None else None
                    percent_change_str = f"({percent_change:.2f}%)" if percent_change is not None else ""
                else:
                    percent_change_str = ""

                html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{before_str}</td>
                <td>{after_str}</td>
                <td>{change_str} {percent_change_str}</td>
            </tr>
    """
            html_content += "</table>"

            # 添加对比图
            html_content += """
        <h2>优化前后可视化对比</h2>
        <div class="image-container">
    """
            for image_name, image_path in comparison_results["images"].items():
                html_content += f"""
            <div class="image-box">
                <h3>{image_name}</h3>
                <img src="{image_path}" alt="{image_name}">
            </div>
    """
            html_content += """
        </div>
    """

        html_content += """
            <h2>优化可视化</h2>
            <div class="image-container">
                <div class="image-box">
                    <h3>三维Pareto前沿</h3>
                    <img src="Visualization/3d_pareto_front.png" alt="三维Pareto前沿">
                </div>
                <div class="image-box">
                    <h3>收敛趋势</h3>
                    <img src="Visualization/convergence_trend.png" alt="收敛趋势">
                </div>
            </div>
        </body>
        </html>
    """

        with open(output_dir / "optimization_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)