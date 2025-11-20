"""
Cloud-Optimized GeoTIFF (COG) Writer for GFM Flood Composites.

This module writes xarray DataArrays as Cloud-Optimized GeoTIFFs with proper
metadata encoding in filenames and optimal COG configuration.
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import xarray as xr
import rioxarray  # For rio accessor
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COGWriter:
    """Write flood extent composites as Cloud-Optimized GeoTIFFs."""

    def __init__(self, output_base_dir: Optional[Path] = None):
        """Initialize COG writer.

        Args:
            output_base_dir: Base directory for output files (default: data/melissa)
        """
        if output_base_dir is None:
            output_base_dir = Path("data/melissa")
        self.output_base_dir = Path(output_base_dir)

    def write_flood_composite(
        self,
        data_array: xr.DataArray,
        iso3: str,
        date: str,
        nodata_value: int = 255,
        overviews: bool = True,
        compression: str = "DEFLATE",
    ) -> Path:
        """Write a flood extent composite as a Cloud-Optimized GeoTIFF.

        Args:
            data_array: xarray DataArray containing flood extent composite
            iso3: ISO3 country code (e.g., 'JAM', 'SOM')
            date: Date string in YYYY-MM-DD format
            nodata_value: NoData value to set (default: 255 for uint8)
            overviews: Whether to create internal overviews (default: True)
            compression: Compression algorithm (default: DEFLATE)

        Returns:
            Path to the written COG file

        Raises:
            ValueError: If inputs are invalid
            IOError: If file writing fails
        """
        # Validate inputs
        iso3 = iso3.upper()
        if len(iso3) != 3:
            raise ValueError(f"ISO3 code must be 3 characters, got: {iso3}")

        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date must be in YYYY-MM-DD format, got: {date}"
            )

        if not isinstance(data_array, xr.DataArray):
            raise ValueError(
                f"data_array must be xarray.DataArray, got: {type(data_array)}"
            )

        # Ensure data array has CRS information
        if data_array.rio.crs is None:
            logger.warning("DataArray missing CRS, setting to EPSG:4326")
            data_array = data_array.rio.write_crs("EPSG:4326")

        # Create output directory
        output_dir = self.output_base_dir / iso3
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Generate filename
        filename = f"gfm_flood_{iso3.lower()}_{date}_composite.tif"
        output_path = output_dir / filename

        logger.info(f"Writing COG to: {output_path}")
        logger.info(f"  Shape: {data_array.shape}")
        logger.info(f"  CRS: {data_array.rio.crs}")
        logger.info(f"  NoData: {nodata_value}")
        logger.info(f"  Compression: {compression}")

        try:
            # Set nodata value
            data_array = data_array.rio.write_nodata(nodata_value)

            # Build COG creation options
            cog_profile = {
                "driver": "GTiff",
                "TILED": "YES",
                "COMPRESS": compression,
                "BLOCKXSIZE": 512,
                "BLOCKYSIZE": 512,
                "BIGTIFF": "IF_SAFER",
            }

            # Add overview options if requested
            if overviews:
                cog_profile["COPY_SRC_OVERVIEWS"] = "YES"

            # Write to COG using rioxarray
            # First write with basic profile
            data_array.rio.to_raster(
                output_path,
                driver="GTiff",
                dtype="uint8",
                tiled=True,
                compress=compression,
                blockxsize=512,
                blockysize=512,
            )

            # If overviews requested, add them using rioxarray's built-in functionality
            if overviews:
                logger.info("Building internal overviews...")
                # Reopen and add overviews
                import rasterio
                from rasterio.enums import Resampling

                with rasterio.open(output_path, "r+") as dst:
                    # Build overviews at 2x, 4x, 8x, 16x
                    overview_levels = [2, 4, 8, 16]
                    dst.build_overviews(overview_levels, Resampling.nearest)
                    dst.update_tags(ns="rio_overview", resampling="nearest")

            # Verify the file was written
            if not output_path.exists():
                raise IOError(f"Failed to write file: {output_path}")

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Successfully wrote COG: {filename}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")

            # Log statistics
            valid_pixels = int((data_array != nodata_value).sum())
            flood_pixels = int((data_array == 1).sum())
            total_pixels = int(data_array.size)
            coverage_pct = 100 * valid_pixels / total_pixels if total_pixels > 0 else 0

            logger.info(f"  Valid pixels: {valid_pixels:,} ({coverage_pct:.2f}% coverage)")
            logger.info(f"  Flood pixels: {flood_pixels:,}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to write COG: {e}")
            # Clean up partial file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Cleaned up partial file: {output_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial file: {cleanup_error}")
            raise IOError(f"Error writing COG: {e}") from e

    def write_multiband_composite(
        self,
        flood_data: xr.DataArray,
        doy_data: xr.DataArray,
        iso3: str,
        date: str,
        flood_nodata: int = 255,
        doy_nodata: int = 0,
        overviews: bool = True,
        compression: str = "DEFLATE",
    ) -> Path:
        """Write a multi-band flood composite (flood extent + DOY) as COG.

        Args:
            flood_data: xarray DataArray with flood extent (0=no flood, 1=flood, 255=nodata)
            doy_data: xarray DataArray with day-of-year provenance (0=no data, 1-366=DOY)
            iso3: ISO3 country code (e.g., 'JAM', 'SOM')
            date: Date string in YYYY-MM-DD format
            flood_nodata: NoData value for flood band (default: 255)
            doy_nodata: NoData value for DOY band (default: 0)
            overviews: Whether to create internal overviews (default: True)
            compression: Compression algorithm (default: DEFLATE)

        Returns:
            Path to the written COG file

        Raises:
            ValueError: If inputs are invalid
            IOError: If file writing fails
        """
        # Validate inputs
        iso3 = iso3.upper()
        if len(iso3) != 3:
            raise ValueError(f"ISO3 code must be 3 characters, got: {iso3}")

        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                f"Date must be in YYYY-MM-DD format, got: {date}"
            )

        if flood_data.shape != doy_data.shape:
            raise ValueError(
                f"Flood and DOY arrays must have same shape. "
                f"Got flood: {flood_data.shape}, DOY: {doy_data.shape}"
            )

        # Create output directory
        output_dir = self.output_base_dir / iso3
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"gfm_flood_{iso3.lower()}_{date}_composite.tif"
        output_path = output_dir / filename

        logger.info(f"Writing multi-band COG to: {output_path}")
        logger.info(f"  Shape: {flood_data.shape}")
        logger.info(f"  CRS: {flood_data.rio.crs}")
        logger.info(f"  Bands: 2 (flood extent + DOY)")
        logger.info(f"  Compression: {compression}")

        try:
            # Stack into multi-band dataset
            import rasterio
            from rasterio.enums import Resampling

            # Ensure CRS is set
            if flood_data.rio.crs is None:
                logger.warning("DataArray missing CRS, setting to EPSG:4326")
                flood_data = flood_data.rio.write_crs("EPSG:4326")
                doy_data = doy_data.rio.write_crs("EPSG:4326")

            # Prepare rasterio profile
            profile = {
                "driver": "GTiff",
                "height": flood_data.shape[0],
                "width": flood_data.shape[1],
                "count": 2,
                "dtype": "uint8",
                "crs": str(flood_data.rio.crs),
                "transform": flood_data.rio.transform(),
                "compress": compression,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }

            # Write multi-band file
            with rasterio.open(output_path, "w", **profile) as dst:
                # Write band 1: flood extent
                dst.write(flood_data.values.astype(np.uint8), 1)
                dst.set_band_description(
                    1, "Flood Extent (0=no flood, 1=flood, 255=nodata)"
                )

                # Write band 2: DOY provenance
                dst.write(doy_data.values.astype(np.uint8), 2)
                dst.set_band_description(
                    2, "Day of Year (0=no observation, 1-255=DOY clamped)"
                )

                # Build overviews if requested
                if overviews:
                    logger.info("Building internal overviews...")
                    overview_levels = [2, 4, 8, 16]
                    dst.build_overviews(overview_levels, Resampling.nearest)
                    dst.update_tags(ns="rio_overview", resampling="nearest")

            # Verify the file was written
            if not output_path.exists():
                raise IOError(f"Failed to write file: {output_path}")

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Successfully wrote multi-band COG: {filename}")
            logger.info(f"  File size: {file_size_mb:.2f} MB")

            # Log statistics
            valid_pixels = int((flood_data != flood_nodata).sum())
            flood_pixels = int((flood_data == 1).sum())
            total_pixels = int(flood_data.size)
            coverage_pct = 100 * valid_pixels / total_pixels if total_pixels > 0 else 0

            logger.info(f"  Valid pixels: {valid_pixels:,} ({coverage_pct:.2f}% coverage)")
            logger.info(f"  Flood pixels: {flood_pixels:,}")

            return output_path

        except Exception as e:
            logger.error(f"Failed to write multi-band COG: {e}")
            # Clean up partial file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                    logger.info(f"Cleaned up partial file: {output_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial file: {cleanup_error}")
            raise IOError(f"Error writing multi-band COG: {e}") from e


def main():
    """Example usage of COG writer."""
    import numpy as np

    logger.info("=" * 60)
    logger.info("COG WRITER EXAMPLE")
    logger.info("=" * 60)

    # Create example data
    logger.info("Creating example flood composite...")
    height, width = 1000, 1000
    flood_data = np.random.choice([0, 1, 255], size=(height, width), p=[0.85, 0.10, 0.05])

    # Create xarray DataArray with spatial reference
    x_coords = np.linspace(-75.0, -74.0, width)
    y_coords = np.linspace(18.5, 17.5, height)

    flood_array = xr.DataArray(
        flood_data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        attrs={"long_name": "GFM Flood Extent"},
    )
    flood_array = flood_array.rio.write_crs("EPSG:4326")

    # Example 1: Single-band COG
    logger.info("\nExample 1: Writing single-band COG...")
    writer = COGWriter()
    try:
        output_path = writer.write_flood_composite(
            data_array=flood_array,
            iso3="JAM",
            date="2025-10-27",
            nodata_value=255,
            overviews=True,
            compression="DEFLATE",
        )
        logger.info(f"✅ Single-band COG written to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to write single-band COG: {e}")

    # Example 2: Multi-band COG (flood + DOY)
    logger.info("\nExample 2: Writing multi-band COG (flood + DOY)...")
    doy_data = np.full((height, width), 255, dtype=np.uint8)  # DOY = 255 (clamped)
    doy_array = xr.DataArray(
        doy_data,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
        attrs={"long_name": "Day of Year"},
    )
    doy_array = doy_array.rio.write_crs("EPSG:4326")

    try:
        output_path = writer.write_multiband_composite(
            flood_data=flood_array,
            doy_data=doy_array,
            iso3="JAM",
            date="2025-10-27",
            overviews=True,
            compression="DEFLATE",
        )
        logger.info(f"✅ Multi-band COG written to: {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to write multi-band COG: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("COG WRITER EXAMPLE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
