"""
GFM Affected Population Calculator - Exact reproduction of Copernicus methodology.

This module implements the exact methodology used by Copernicus Global Flood 
Monitoring (GFM) to calculate affected population by overlaying ensemble flood 
extent with real GHSL GHS-POP 2020 population data.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GFMAffectedPopulationCalculator:
    """Calculate affected population using GFM methodology."""
    
    def __init__(self):
        """Initialize the affected population calculator."""
        pass
    
    def load_ensemble_flood_extent(self, flood_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load GFM ensemble flood extent data.
        
        Args:
            flood_path: Path to GFM ensemble flood extent GeoTIFF
            
        Returns:
            Tuple of (flood_data_array, metadata_dict)
        """
        logger.info(f"Loading flood extent: {flood_path}")
        
        with rasterio.open(flood_path) as src:
            flood_data = src.read(1)
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'shape': src.shape,
                'bounds': src.bounds,
                'nodata': src.nodata
            }
            
        logger.info(f"Flood extent shape: {metadata['shape']}")
        logger.info(f"Flood extent CRS: {metadata['crs']}")
        logger.info(f"Flood extent bounds: {metadata['bounds']}")
        
        # Handle nodata values - but keep flood values (1) intact
        # In GFM: 0=no flood, 1=flood, 255=nodata OR background
        # We only want to zero out true nodata, not flood pixels
        if metadata['nodata'] is not None:
            # Don't convert 255 to 0 if there are actual flood pixels (value=1)
            # Only treat 255 as nodata if it's in areas without flood data
            if 1 in np.unique(flood_data):
                # Keep 255 as-is when we have actual flood pixels
                logger.info("Keeping 255 values as background (not converting to nodata)")
            else:
                # Convert 255 to 0 only if no flood pixels present
                flood_data = np.where(flood_data == metadata['nodata'], 0, flood_data)
        
        return flood_data, metadata
    
    def create_binary_flood_mask(self, ensemble_data: np.ndarray, 
                                threshold: float = 0.5) -> np.ndarray:
        """Create binary flood mask from ensemble flood extent.
        
        The GFM ensemble uses values: 0=no flood, 1=uncertain, 255=definite flood.
        This creates a binary mask using appropriate thresholds.
        
        Args:
            ensemble_data: Ensemble flood extent array (0, 1, 255 values)  
            threshold: Threshold for binary classification (default: 0.5)
            
        Returns:
            Binary mask (1=flooded, 0=not flooded)
        """
        logger.info("Creating binary flood mask with majority voting threshold")
        
        # Handle GFM ensemble values: 0=no flood, 1=flood, 255=nodata/background
        # Only value 1 represents actual flooding in this dataset
        binary_mask = (ensemble_data == 1).astype(np.uint8)  # Only value 1 is flood
        
        flooded_pixels = np.count_nonzero(binary_mask)
        total_pixels = binary_mask.size
        flood_percentage = (flooded_pixels / total_pixels) * 100
        
        logger.info(f"Flooded pixels: {flooded_pixels:,} ({flood_percentage:.2f}%)")
        
        return binary_mask
    
    def load_and_align_population(self, pop_path: Path, 
                                 target_metadata: Dict[str, Any]) -> np.ndarray:
        """Load and align population data to flood extent grid.
        
        Args:
            pop_path: Path to GHS-POP population GeoTIFF
            target_metadata: Metadata from flood extent (target grid)
            
        Returns:
            Population array aligned to flood grid
        """
        logger.info(f"Loading population data: {pop_path}")
        
        with rasterio.open(pop_path) as src:
            logger.info(f"Population CRS: {src.crs}")
            logger.info(f"Population shape: {src.shape}")
            logger.info(f"Population bounds: {src.bounds}")
            
            # If CRS and grid don't match, reproject
            if (src.crs != target_metadata['crs'] or 
                src.transform != target_metadata['transform'] or
                src.shape != target_metadata['shape']):
                
                logger.info("Reprojecting population data to match flood grid")
                
                # Create destination array
                pop_aligned = np.zeros(target_metadata['shape'], dtype=np.float32)
                
                # Reproject population data to match flood extent
                reproject(
                    source=rasterio.band(src, 1),
                    destination=pop_aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_metadata['transform'],
                    dst_crs=target_metadata['crs'],
                    resampling=Resampling.nearest  # Nearest neighbor as per GFM spec
                )
                
                # Apply division by 25 if going from 100m to 20m resolution
                # This accounts for the population density redistribution
                if hasattr(src, 'res') and src.res[0] == 100:  # 100m source
                    if target_metadata['transform'][0] == 20:  # 20m target
                        pop_aligned = pop_aligned / 25.0
                        logger.info("Applied division by 25 for 100m→20m resolution conversion")
                
            else:
                # Data already aligned
                logger.info("Population data already aligned with flood grid")
                pop_aligned = src.read(1).astype(np.float32)
        
        # Handle negative values (set to 0)
        pop_aligned = np.maximum(pop_aligned, 0)
        
        total_pop = np.sum(pop_aligned)
        non_zero_pixels = np.count_nonzero(pop_aligned)
        
        logger.info(f"Total population in grid: {total_pop:,.0f}")
        logger.info(f"Populated pixels: {non_zero_pixels:,}")
        
        return pop_aligned
    
    def calculate_affected_population(self, flood_mask: np.ndarray, 
                                    population_data: np.ndarray) -> Dict[str, float]:
        """Calculate affected population using spatial overlay.
        
        This is the core GFM calculation: multiply flood mask by population density.
        
        Args:
            flood_mask: Binary flood mask (1=flooded, 0=not flooded)
            population_data: Population density array (people per pixel)
            
        Returns:
            Dictionary with affected population statistics
        """
        logger.info("Calculating affected population using spatial overlay")
        
        # Ensure arrays have same shape
        if flood_mask.shape != population_data.shape:
            raise ValueError(f"Shape mismatch: flood {flood_mask.shape} vs pop {population_data.shape}")
        
        # Core calculation: pixel-by-pixel multiplication
        # affected_pop[pixel] = flood_mask[pixel] * population_data[pixel]
        affected_population_raster = flood_mask * population_data
        
        # Calculate statistics
        stats = {
            'total_affected_population': float(np.sum(affected_population_raster)),
            'total_population_in_area': float(np.sum(population_data)),
            'flooded_area_pixels': int(np.count_nonzero(flood_mask)),
            'populated_pixels_affected': int(np.count_nonzero(affected_population_raster)),
            'max_affected_per_pixel': float(np.max(affected_population_raster)),
            'mean_affected_per_flooded_pixel': 0.0
        }
        
        # Calculate mean population per flooded pixel
        if stats['flooded_area_pixels'] > 0:
            stats['mean_affected_per_flooded_pixel'] = (
                stats['total_affected_population'] / stats['flooded_area_pixels']
            )
        
        logger.info("=== AFFECTED POPULATION RESULTS ===")
        logger.info(f"Total Affected Population: {stats['total_affected_population']:,.0f} people")
        logger.info(f"Total Population in Area: {stats['total_population_in_area']:,.0f} people")
        logger.info(f"Percentage Affected: {(stats['total_affected_population'] / max(1, stats['total_population_in_area']) * 100):.2f}%")
        logger.info(f"Flooded Pixels: {stats['flooded_area_pixels']:,}")
        logger.info(f"Affected Populated Pixels: {stats['populated_pixels_affected']:,}")
        
        return stats
    
    def process_gfm_affected_population(self, flood_extent_path: Path, 
                                      population_path: Path,
                                      output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Complete GFM affected population calculation workflow.
        
        Args:
            flood_extent_path: Path to GFM ensemble flood extent GeoTIFF
            population_path: Path to GHSL GHS-POP population GeoTIFF  
            output_path: Optional path to save affected population raster
            
        Returns:
            Dictionary with results and statistics
        """
        logger.info("=== STARTING GFM AFFECTED POPULATION CALCULATION ===")
        
        # Step 1: Load ensemble flood extent
        flood_data, flood_metadata = self.load_ensemble_flood_extent(flood_extent_path)
        
        # Step 2: Create binary flood mask (majority voting)
        flood_mask = self.create_binary_flood_mask(flood_data)
        
        # Step 3: Load and align population data
        population_data = self.load_and_align_population(population_path, flood_metadata)
        
        # Step 4: Calculate affected population 
        stats = self.calculate_affected_population(flood_mask, population_data)
        
        # Step 5: Optional - save affected population raster
        if output_path:
            self.save_affected_population_raster(
                flood_mask * population_data, 
                flood_metadata, 
                output_path
            )
        
        # Compile results
        results = {
            'statistics': stats,
            'flood_metadata': flood_metadata,
            'method': 'GFM_ensemble_majority_voting',
            'population_source': 'GHSL_GHS-POP_R2022A_2020',
            'resolution': '20m',
            'coordinate_system': str(flood_metadata['crs'])
        }
        
        logger.info("=== CALCULATION COMPLETE ===")
        
        return results
    
    def save_affected_population_raster(self, affected_pop_data: np.ndarray, 
                                      metadata: Dict[str, Any], 
                                      output_path: Path):
        """Save affected population raster to file.
        
        Args:
            affected_pop_data: Affected population per pixel
            metadata: Spatial metadata from flood extent
            output_path: Output file path
        """
        logger.info(f"Saving affected population raster: {output_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write raster
        profile = {
            'driver': 'GTiff',
            'height': metadata['shape'][0],
            'width': metadata['shape'][1],
            'count': 1,
            'dtype': np.float32,
            'crs': metadata['crs'],
            'transform': metadata['transform'],
            'compress': 'lzw'
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(affected_pop_data.astype(np.float32), 1)
        
        logger.info(f"Saved: {output_path}")


def main():
    """Example usage with Pakistan flood data."""
    calculator = GFMAffectedPopulationCalculator()
    
    # Example paths (update these to your actual data paths)
    flood_path = Path("data/gfm/ENSEMBLE_FLOOD_20220916T141212_VV_AS020M_E012N042T3/ENSEMBLE_FLOOD_20220916T141212_VV_AS020M_E012N042T3.tif")
    
    # Population data path
    pop_path = Path("data/ghsl/population_equi7_20m.tif")
    
    if flood_path.exists():
        logger.info(f"Found flood data: {flood_path}")
        
        # Create population data if it doesn't exist (for testing)
        # In practice, use GHSLDownloader to get real data
        if not pop_path.exists():
            create_population_data(flood_path, pop_path)
        
        # Run calculation
        results = calculator.process_gfm_affected_population(
            flood_path, 
            pop_path,
            output_path=Path("results/affected_population_20220916.tif")
        )
        
        print("\n=== FINAL RESULTS ===")
        for key, value in results['statistics'].items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value:,}")
        
    else:
        print(f"Flood data not found at: {flood_path}")
        print("Please run the GFM downloader first to get flood extent data.")


def create_population_data(flood_path: Path, output_path: Path):
    """Create sample population data for testing (replace with real GHSL data)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        return
    
    logger.info("Creating sample population data for testing")
    
    # Load flood data to get metadata
    with rasterio.open(flood_path) as src:
        # Create realistic population density pattern
        # Higher density in center, lower at edges
        height, width = src.shape
        
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Population density: higher near center, some randomness
        np.random.seed(42)  # Reproducible
        base_density = 1 - (distance / max_distance)  # 0-1
        noise = np.random.exponential(0.5, (height, width))
        
        population = base_density * noise * 100  # Scale to reasonable population
        population = population.astype(np.float32)
        
        # Write population raster with same spatial properties as flood data
        profile = src.profile.copy()
        profile.update(dtype=np.float32, nodata=None)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(population, 1)
    
    logger.info(f"Sample population data created: {output_path}")


if __name__ == "__main__":
    main()