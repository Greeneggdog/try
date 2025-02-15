from facility_optimizer import FacilityOptimizer
import logging
from pathlib import Path

def main():
    # Initialize optimizer with custom logging level
    optimizer = FacilityOptimizer(log_level=logging.INFO)
    
    # Define parameters
    params = {
        "existing_facilities": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/facilities/existing_facilities.json",
        "population_raster": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/MedicalFacility_20250210_191914/population_projected.tif",
        "accessibility_raster": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/Accessibility_2025-02-10_115835/accessibility.tif",
        "land_cost_raster": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/MedicalFacility_20250210_191914/cost_projected.tif",
        "new_facility_count": 5,
        "service_radius": 5000,
        "output_folder": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/Output",
        "population_size": 50,
        "generations": 20,
        "underserved_threshold_percentile": 60,
        "num_candidates": 200,
        "shanghai_boundary": "D:/Desktop/Study_Material/Thesis_Associated_Resources/Code/facilities/shanghai_Dissolve.json"
    }
    
    # Run optimization
    try:
        result_dir = optimizer.execute(params)
        print(f"Optimization completed. Results saved to: {result_dir}")
    except Exception as e:
        print(f"Optimization failed: {str(e)}")

if __name__ == "__main__":
    main()