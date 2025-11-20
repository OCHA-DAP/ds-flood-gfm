"""
GHSL GHS-POP data downloader for population overlay analysis.

This module provides functionality to download GHS-POP 2020 population data
from the official Copernicus GHSL sources for use in flood impact analysis.
"""

import os
import requests
from pathlib import Path
from typing import Tuple, Optional
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GHSLDownloader:
    """Download and process GHSL GHS-POP population data."""
    
    # GHSL GHS-POP R2022A download URLs (100m resolution, 2020)
    BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2022A/GHS_POP_E{year}_GLOBE_R2022A_54009_100/V1-0/"
    
    # Pakistan region tiles (approximate coverage)
    # These cover the major populated areas of Pakistan
    PAKISTAN_TILES = [
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R5_C18",  # Northern Pakistan
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R5_C19",  # Northeast Pakistan  
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R6_C18",  # Central Pakistan
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R6_C19",  # Central-East Pakistan
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R7_C18",  # Southern Pakistan
        "GHS_POP_E2020_GLOBE_R2022A_54009_100_V1_0_R7_C19",  # Southeast Pakistan
    ]
    
    def __init__(self, download_dir: str = "./data/ghsl"):
        """Initialize GHSL downloader.
        
        Args:
            download_dir: Directory to store downloaded GHSL data
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_tile(self, tile_name: str, year: int = 2020) -> Path:
        """Download a specific GHSL tile.
        
        Args:
            tile_name: Name of the tile to download
            year: Year of the population data (default: 2020)
            
        Returns:
            Path to downloaded file
        """
        url = f"{self.BASE_URL.format(year=year)}{tile_name}.zip"
        local_path = self.download_dir / f"{tile_name}.zip"
        
        if local_path.exists():
            logger.info(f"File already exists: {local_path}")
            return local_path
        
        logger.info(f"Downloading {tile_name} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=tile_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded: {local_path}")
            return local_path
            
        except requests.RequestException as e:
            logger.error(f"Error downloading {tile_name}: {e}")
            if local_path.exists():
                local_path.unlink()
            raise
    
    def extract_tile(self, zip_path: Path) -> Path:
        """Extract GHSL tile from zip file.
        
        Args:
            zip_path: Path to zip file
            
        Returns:
            Path to extracted TIF file
        """
        import zipfile
        
        extract_dir = zip_path.parent / zip_path.stem
        extract_dir.mkdir(exist_ok=True)
        
        tif_path = None
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.tif'):
                    zip_ref.extract(file_name, extract_dir)
                    tif_path = extract_dir / file_name
                    break
        
        if not tif_path:
            raise ValueError(f"No TIF file found in {zip_path}")
        
        logger.info(f"Extracted: {tif_path}")
        return tif_path
    
    def download_pakistan_population(self, year: int = 2020) -> list[Path]:
        """Download all Pakistan population tiles.
        
        Args:
            year: Year of population data
            
        Returns:
            List of paths to extracted TIF files
        """
        tif_files = []
        
        for tile_name in self.PAKISTAN_TILES:
            try:
                # Download and extract tile
                zip_path = self.download_tile(tile_name, year)
                tif_path = self.extract_tile(zip_path)
                tif_files.append(tif_path)
                
            except Exception as e:
                logger.warning(f"Failed to process tile {tile_name}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(tif_files)} Pakistan population tiles")
        return tif_files
    
    def reproject_to_equi7(self, ghsl_tif_path: Path, target_crs: str, 
                          target_bounds: Tuple[float, float, float, float], 
                          target_resolution: float = 20.0) -> Optional[Path]:
        """Reproject GHSL data to Equi7Grid coordinate system.
        
        Args:
            ghsl_tif_path: Path to GHSL TIF file
            target_crs: Target CRS (e.g., 'EPSG:27703' for Asia Equi7Grid)
            target_bounds: Target bounds (left, bottom, right, top)
            target_resolution: Target resolution in meters (default: 20m)
            
        Returns:
            Path to reprojected file
        """
        output_path = ghsl_tif_path.parent / f"{ghsl_tif_path.stem}_equi7_20m.tif"
        
        if output_path.exists():
            logger.info(f"Reprojected file already exists: {output_path}")
            return output_path
        
        logger.info(f"Reprojecting {ghsl_tif_path} to {target_crs}")
        
        try:
            with rasterio.open(ghsl_tif_path) as src:
                # Calculate transform for target grid
                left, bottom, right, top = target_bounds
                width = int((right - left) / target_resolution)
                height = int((top - bottom) / target_resolution)
                
                transform = rasterio.Affine(target_resolution, 0.0, left,
                                          0.0, -target_resolution, top)
                
                # Reproject to target CRS
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'dtype': 'float32'  # Use float32 for population counts
                })
                
                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
                    
                    # Divide by 25 to account for resolution change (100m² to 20m²)
                    # Each 100m pixel is divided among 25 20m pixels
                    data = dst.read(1)
                    data = np.where(data > 0, data / 25.0, 0)
                    dst.write(data, 1)
            
            logger.info(f"Reprojected file saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error reprojecting {ghsl_tif_path}: {e}")
            if output_path.exists():
                output_path.unlink()
            return None
    
    def get_population_for_bounds(self, bounds: Tuple[float, float, float, float], 
                                 crs: str = "EPSG:27703") -> Optional[Path]:
        """Get population data for specific bounds.
        
        Args:
            bounds: Bounds in target CRS (left, bottom, right, top)  
            crs: Target coordinate reference system
            
        Returns:
            Path to processed population raster for the bounds
        """
        # Download Pakistan tiles if not already done
        tif_files = self.download_pakistan_population()
        
        if not tif_files:
            logger.error("No population tiles downloaded")
            return None
        
        # For now, process the first available tile that intersects
        # In a full implementation, we'd mosaic all relevant tiles
        for tif_path in tif_files:
            try:
                reprojected_path = self.reproject_to_equi7(
                    tif_path, crs, bounds, target_resolution=20.0
                )
                if reprojected_path:
                    return reprojected_path
            except Exception as e:
                logger.warning(f"Failed to reproject {tif_path}: {e}")
                continue
        
        logger.error("No population data could be processed for the given bounds")
        return None


def main():
    """Example usage of GHSLDownloader."""
    # Initialize downloader
    downloader = GHSLDownloader()
    
    # Example bounds from our GFM data
    # Bounds: (1200000.0, 4200000.0, 1500000.0, 4500000.0)
    gfm_bounds = (1200000.0, 4200000.0, 1500000.0, 4500000.0)
    target_crs = "EPSG:27703"  # Asia Equi7Grid
    
    try:
        # Get population data for the GFM tile bounds
        pop_raster = downloader.get_population_for_bounds(gfm_bounds, target_crs)
        
        if pop_raster:
            print(f"Population raster ready: {pop_raster}")
            
            # Check the data
            with rasterio.open(pop_raster) as src:
                print(f"Population raster shape: {src.shape}")
                print(f"Population raster CRS: {src.crs}")
                print(f"Population raster bounds: {src.bounds}")
                
                # Sample some data
                sample = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                print(f"Population range: {sample.min():.2f} to {sample.max():.2f}")
                print(f"Non-zero pixels: {np.count_nonzero(sample)} / {sample.size}")
        else:
            print("Failed to get population data")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()