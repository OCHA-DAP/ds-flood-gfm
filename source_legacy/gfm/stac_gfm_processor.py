"""
STAC-based GFM Affected Population Calculator

Implements the official GFM methodology for calculating affected population
using STAC API access to raw flood extent data.

Official GFM Methodology:
1. Flood Detection: Binary ensemble (2 out of 3 algorithms agree) → value = 1
2. Population Data: GHS-POP R2023A, 100m resolution, year 2020
3. Calculation: Direct pixel-by-pixel multiplication: flood_mask × population
4. Resolution: Flood at 20m, population at 100m (resampled with nearest neighbor)

Reference:
- GFM Product User Manual: https://extwiki.eodc.eu/gfm_assets/gfm_pum_v20231005_compressed.pdf
- Population Layer: https://global-flood.emergency.copernicus.eu/news/134-release-of-the-updated-potentially-affected-population-layer-in-gfm/
"""

from typing import Union, List, Tuple, Dict
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import rioxarray
import stackstac
from pystac_client import Client
import geopandas as gpd
import ocha_stratus as stratus


class STACGFMProcessor:
    """
    Process GFM flood data from STAC API following official GFM methodology.
    """

    def __init__(
        self,
        stac_url: str = "https://stac.eodc.eu/api/v1",
        population_blob: str = "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif",
    ):
        """
        Initialize the STAC GFM processor.

        Parameters
        ----------
        stac_url : str
            EODC STAC API endpoint
        population_blob : str
            Azure blob path to GHS-POP data (should match GFM's population layer)
        """
        self.stac_url = stac_url
        self.population_blob = population_blob
        self.client = None
        self.population_data = None

    def connect_stac(self):
        """Connect to STAC API"""
        if self.client is None:
            self.client = Client.open(self.stac_url)
        return self.client

    def load_population(self, bbox: Tuple[float, float, float, float]) -> xr.DataArray:
        """
        Load GHS-POP population data for area of interest.

        Parameters
        ----------
        bbox : tuple
            Bounding box (minx, miny, maxx, maxy) in EPSG:4326

        Returns
        -------
        xr.DataArray
            Population density data clipped to bbox
        """
        print(f"  Loading GHS-POP R2023A (2020) from Azure...")
        da_global = stratus.open_blob_cog(
            self.population_blob, container_name="raster"
        ).squeeze(drop=True)

        # Clip to bbox
        da_clip = da_global.rio.clip_box(*bbox)

        # Ensure CRS
        if da_clip.rio.crs is None:
            da_clip = da_clip.rio.write_crs("EPSG:4326")

        # Mask nodata values
        if da_clip.rio.nodata is not None:
            da_clip = da_clip.where(da_clip != da_clip.rio.nodata)

        print(f"  ✅ Population data loaded: {da_clip.shape}")
        return da_clip

    def search_gfm_data(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
    ) -> List:
        """
        Search for GFM flood extent data in STAC catalog.

        Parameters
        ----------
        bbox : tuple
            Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format

        Returns
        -------
        list
            List of STAC items
        """
        client = self.connect_stac()

        print(f"  Searching STAC for GFM data ({start_date} to {end_date})...")
        search = client.search(
            collections=["GFM"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
        )

        items = list(search.items())
        print(f"  ✅ Found {len(items)} STAC items")

        return items

    def create_flood_cube(
        self,
        items: List,
        bbox: Tuple[float, float, float, float],
        resolution: int = 100,
    ) -> xr.DataArray:
        """
        Create a data cube from STAC items.

        Parameters
        ----------
        items : list
            STAC items from search
        bbox : tuple
            Bounding box to clip to
        resolution : int
            Resolution in meters (default 100 to match GHS-POP)

        Returns
        -------
        xr.DataArray
            Data cube of flood extent
        """
        print(f"  Creating stackstac data cube...")

        # Create stack (let stackstac determine bounds automatically)
        stack = stackstac.stack(items, epsg=4326)

        # Select flood extent band
        stack_flood = stack.sel(band="ensemble_flood_extent")

        # Clip to bbox
        print(f"  Clipping to bounding box...")
        stack_flood_clipped = stack_flood.sel(
            x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1])
        )

        print(f"  ✅ Stack shape: {stack_flood_clipped.shape}")
        return stack_flood_clipped

    def composite_daily_max(self, stack_flood: xr.DataArray) -> xr.DataArray:
        """
        Composite multiple observations per day using max value.

        Parameters
        ----------
        stack_flood : xr.DataArray
            Flood extent data cube

        Returns
        -------
        xr.DataArray
            Daily max composited flood extent
        """
        print(f"  Computing daily max composite...")

        # Group by date and take max
        stack_flood_max = stack_flood.groupby("time.date").max()

        # Rename dimension back to 'time'
        stack_flood_max = stack_flood_max.rename({"date": "time"})
        stack_flood_max["time"] = stack_flood_max.time.astype("datetime64[ns]")

        print(f"  ✅ Composited to {len(stack_flood_max.time)} dates")
        return stack_flood_max

    def apply_gfm_threshold(self, flood_data: xr.DataArray) -> xr.DataArray:
        """
        Apply GFM's official binary flood threshold.

        GFM Methodology: ensemble_flood_extent == 1 means flood
        (2 out of 3 algorithms agree)

        Parameters
        ----------
        flood_data : xr.DataArray
            Raw flood extent data

        Returns
        -------
        xr.DataArray
            Binary flood mask (0 or 1)
        """
        # GFM official threshold: exactly 1 = flood
        flood_binary = (flood_data == 1).astype(int)

        # Preserve CRS
        if flood_data.rio.crs is not None:
            flood_binary = flood_binary.rio.write_crs(flood_data.rio.crs)

        return flood_binary

    def calculate_affected_population(
        self,
        flood_binary: xr.DataArray,
        population: xr.DataArray,
        aoi_geometry: gpd.GeoSeries = None,
    ) -> Tuple[xr.DataArray, float]:
        """
        Calculate affected population using GFM methodology.

        Parameters
        ----------
        flood_binary : xr.DataArray
            Binary flood mask (0 or 1)
        population : xr.DataArray
            Population density data
        aoi_geometry : gpd.GeoSeries, optional
            If provided, clip final result to this geometry

        Returns
        -------
        tuple
            (affected_population_raster, total_affected_count)
        """
        # Reproject flood to match population grid (nearest neighbor per GFM spec)
        flood_resampled = flood_binary.rio.reproject_match(
            population, resampling=0  # 0 = nearest neighbor
        )

        # Multiply flood mask by population (GFM methodology)
        affected_pop = flood_resampled * population

        # Clip to AOI geometry if provided
        if aoi_geometry is not None:
            affected_pop = affected_pop.rio.clip(
                aoi_geometry.values, population.rio.crs, drop=False
            )

        # Calculate total
        total_affected = float(affected_pop.sum(skipna=True).values)

        return affected_pop, total_affected

    def process_date_range(
        self,
        aoi: Union[gpd.GeoDataFrame, Tuple[float, float, float, float]],
        start_date: str,
        end_date: str,
        resolution: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Dict]:
        """
        Process GFM data for a date range following official methodology.

        Parameters
        ----------
        aoi : gpd.GeoDataFrame or tuple
            Area of interest (GeoDataFrame or bbox)
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        resolution : int
            Processing resolution in meters (default 100 to match GHS-POP)
        verbose : bool
            Print progress messages

        Returns
        -------
        dict
            Results dictionary with dates as keys:
            {
                "YYYY-MM-DD": {
                    "affected_population": int,
                    "flooded_cells": int,
                    "affected_pop_raster": xr.DataArray
                }
            }
        """
        print("=" * 80)
        print("STAC GFM Processor - Official Methodology Implementation")
        print("=" * 80)

        # Get bbox
        if isinstance(aoi, gpd.GeoDataFrame):
            bbox = tuple(aoi.total_bounds)
            aoi_geometry = aoi.geometry
        else:
            bbox = aoi
            aoi_geometry = None

        # 1. Load population data
        print("\n1. Loading Population Data")
        population = self.load_population(bbox)

        # Clip to AOI geometry if provided
        if aoi_geometry is not None:
            print("  Clipping population to AOI boundary...")
            population = population.rio.clip(aoi_geometry.values, population.rio.crs)
            print(f"  ✅ Population clipped: {(~population.isnull()).sum().values:,} cells")

        # 2. Search for GFM data
        print("\n2. Searching STAC Catalog")
        items = self.search_gfm_data(bbox, start_date, end_date)

        if len(items) == 0:
            print("  ⚠️  No data found for date range")
            return {}

        # 3. Create flood cube
        print("\n3. Creating Flood Data Cube")
        stack_flood = self.create_flood_cube(items, bbox, resolution)

        # 4. Daily composite
        print("\n4. Daily Compositing")
        flood_daily = self.composite_daily_max(stack_flood)

        # 5. Process each date
        print("\n5. Processing Each Date")
        print("-" * 80)

        results = {}

        for i, time_step in enumerate(flood_daily.time):
            date_str = str(time_step.values)[:10]

            if verbose:
                print(f"\n📅 {date_str} (date {i+1}/{len(flood_daily.time)})")

            try:
                # Get flood data for this date
                if verbose:
                    print("  Computing flood data from EODC...")
                flood_data = flood_daily.sel(time=time_step)
                flood_data_computed = flood_data.compute()

                # Apply GFM threshold
                if verbose:
                    print("  Applying GFM binary threshold (==1)...")
                flood_binary = self.apply_gfm_threshold(flood_data_computed)

                flooded_cells = int((flood_binary > 0).sum().values)

                if verbose:
                    print(f"    Flooded cells: {flooded_cells:,}")

                # Calculate affected population
                if verbose:
                    print("  Calculating affected population...")

                affected_pop_raster, total_affected = self.calculate_affected_population(
                    flood_binary, population, aoi_geometry
                )

                # Store results
                results[date_str] = {
                    "affected_population": int(total_affected),
                    "flooded_cells": flooded_cells,
                    "affected_pop_raster": affected_pop_raster,
                }

                if verbose:
                    print(f"  ✅ {int(total_affected):,} people affected")

            except Exception as e:
                print(f"  ⚠️  ERROR: {e}")
                continue

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        if results:
            print(f"\n{'Date':<12} {'Flooded Cells':>15} {'Affected Population':>20}")
            print("-" * 50)
            for date_str, result in sorted(results.items()):
                print(
                    f"{date_str:<12} {result['flooded_cells']:>15,} {result['affected_population']:>20,}"
                )
        else:
            print("No results to display")

        return results

    def save_results(
        self,
        results: Dict[str, Dict],
        output_dir: Union[str, Path],
        save_rasters: bool = False,
    ):
        """
        Save results to disk.

        Parameters
        ----------
        results : dict
            Results from process_date_range()
        output_dir : str or Path
            Output directory
        save_rasters : bool
            If True, save affected population rasters as GeoTIFF
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary CSV
        import pandas as pd

        summary_data = []
        for date_str, result in sorted(results.items()):
            summary_data.append(
                {
                    "date": date_str,
                    "affected_population": result["affected_population"],
                    "flooded_cells": result["flooded_cells"],
                }
            )

        df = pd.DataFrame(summary_data)
        csv_path = output_dir / "affected_population_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Summary saved to: {csv_path}")

        # Save rasters if requested
        if save_rasters:
            raster_dir = output_dir / "rasters"
            raster_dir.mkdir(exist_ok=True)

            for date_str, result in results.items():
                raster_path = raster_dir / f"affected_population_{date_str}.tif"
                result["affected_pop_raster"].rio.to_raster(raster_path)

            print(f"✅ Rasters saved to: {raster_dir}")


# Example usage
if __name__ == "__main__":
    import sys
    from fsspec.implementations.http import HTTPFileSystem

    print("=" * 80)
    print("STAC GFM Processor - Example Usage")
    print("=" * 80)

    # Load CM004 boundary
    print("\nLoading CM004 boundary...")
    GLOBAL_ADM1 = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"
    ISO3 = "CMR"
    filesystem = HTTPFileSystem()
    filters = [("iso_3", "=", ISO3)]
    gdf = gpd.read_parquet(GLOBAL_ADM1, filesystem=filesystem, filters=filters)
    gdf_cm004 = gdf[gdf.adm1_src == "CM004"]

    # Initialize processor
    processor = STACGFMProcessor()

    # Process date range
    results = processor.process_date_range(
        aoi=gdf_cm004,
        start_date="2024-10-18",
        end_date="2024-10-18",
        resolution=100,
        verbose=True,
    )

    # Save results
    output_dir = Path("data/stac_gfm_results")
    processor.save_results(results, output_dir, save_rasters=True)

    print("\n" + "=" * 80)
    print("✅ Processing complete!")
    print("=" * 80)

    sys.exit(0)
