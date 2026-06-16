"""
Download Latest GFM Data for Countries

This module provides functionality to query the GFM STAC API and create daily
composites of flood monitoring data for a given country (by ISO3 code).

Features:
- Queries GFM STAC API (https://stac.eodc.eu/api/v1) for last 12 days
- Finds the most recent 3 dates with actual data coverage
- Creates daily composites by merging overlapping tiles
- Writes composites as Cloud-Optimized GeoTIFFs (COGs)
- Loads country boundaries from FieldMaps.io humanitarian data

Example:
    >>> from ds_flood_gfm.download_latest import process_latest_gfm
    >>> output_paths = process_latest_gfm("JAM")  # Jamaica
    >>> print(f"Created {len(output_paths)} daily COGs")
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from pystac_client import Client
from pystac import ItemCollection
import geopandas as gpd
from fsspec.implementations.http import HTTPFileSystem
import stackstac
import xarray as xr
import numpy as np
from shapely.geometry import Point
from ds_flood_gfm.write_cog import COGWriter


class LatestGFMDownloader:
    """
    Download latest GFM flood data for countries using STAC API.

    This class provides methods to query the GFM STAC catalog for recent flood
    monitoring data based on country boundaries.
    """

    # EODC STAC API endpoint
    STAC_ENDPOINT = "https://stac.eodc.eu/api/v1"

    # GFM Collection ID
    COLLECTION_ID = "GFM"

    # FieldMaps.io humanitarian boundaries endpoint
    FIELDMAPS_ADM1_URL = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"

    def __init__(self, stac_url: str = None):
        """
        Initialize the Latest GFM Downloader.

        Parameters
        ----------
        stac_url : str, optional
            STAC API endpoint URL. Defaults to EODC STAC endpoint.
        """
        self.stac_url = stac_url or self.STAC_ENDPOINT
        self.client = None

    def _connect_stac(self) -> Client:
        """
        Connect to STAC API.

        Returns
        -------
        Client
            PySTAC Client instance

        Raises
        ------
        RuntimeError
            If connection to STAC API fails
        """
        if self.client is None:
            try:
                print(f"Connecting to STAC API: {self.stac_url}")
                self.client = Client.open(self.stac_url)
                print("Successfully connected to STAC API")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to STAC API: {e}")

        return self.client

    def load_country_boundary(self, iso3_code: str) -> gpd.GeoDataFrame:
        """
        Load country boundary from FieldMaps.io humanitarian data.

        This uses edge-matched humanitarian boundaries from FieldMaps.io, based on
        UN Common Operational Datasets (COD) when available.

        Parameters
        ----------
        iso3_code : str
            ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')

        Returns
        -------
        gpd.GeoDataFrame
            Country boundary GeoDataFrame with geometry

        Raises
        ------
        ValueError
            If no data found for the given ISO3 code
        """
        print(f"Loading country boundary for {iso3_code}...")

        try:
            filesystem = HTTPFileSystem()
            filters = [("iso_3", "=", iso3_code)]

            # Read ADM1 data for the country
            gdf = gpd.read_parquet(
                self.FIELDMAPS_ADM1_URL,
                filesystem=filesystem,
                filters=filters
            )

            if len(gdf) == 0:
                raise ValueError(f"No data found for ISO3 code: {iso3_code}")

            print(f"Successfully loaded boundary: {len(gdf)} ADM1 units")
            return gdf

        except Exception as e:
            raise ValueError(f"Error loading country boundary for {iso3_code}: {e}")

    def get_bbox_from_gdf(self, gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
        """
        Extract bounding box from GeoDataFrame.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame with geometry

        Returns
        -------
        tuple
            Bounding box as (minx, miny, maxx, maxy)
        """
        bbox = tuple(gdf.total_bounds)
        print(f"Bounding box: {bbox}")
        return bbox

    def get_latest_date_range(self, days: int = 12) -> Tuple[str, str]:
        """
        Calculate date range for latest N days.

        Parameters
        ----------
        days : int, optional
            Number of days to look back from today (default: 12)

        Returns
        -------
        tuple
            (start_date, end_date) as ISO format strings (YYYY-MM-DD)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"Date range: {start_str} to {end_str} ({days} days)")
        return start_str, end_str

    def get_recent_dates_from_items(self, items: ItemCollection, num_dates: int = 3) -> List[str]:
        """
        Extract the most recent N unique dates from STAC items.

        Parameters
        ----------
        items : ItemCollection
            STAC ItemCollection with search results
        num_dates : int, optional
            Number of most recent dates to return (default: 3)

        Returns
        -------
        list
            List of date strings in YYYY-MM-DD format, sorted newest to oldest
        """
        if len(items) == 0:
            print("No items found in collection")
            return []

        # Extract unique dates from items
        dates = set()
        for item in items:
            if item.datetime:
                date_str = str(item.datetime)[:10]
                dates.add(date_str)

        # Sort dates in descending order (newest first)
        sorted_dates = sorted(dates, reverse=True)

        # Return the most recent N dates
        recent_dates = sorted_dates[:num_dates]

        print(f"Found {len(sorted_dates)} unique dates in collection")
        print(f"Most recent {num_dates} dates: {recent_dates}")

        return recent_dates

    def search_gfm_data(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_items: Optional[int] = None
    ) -> ItemCollection:
        """
        Search for GFM flood data in STAC catalog.

        Parameters
        ----------
        bbox : tuple
            Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        max_items : int, optional
            Maximum number of items to return. If None, returns all items.

        Returns
        -------
        ItemCollection
            STAC ItemCollection with search results

        Raises
        ------
        RuntimeError
            If search fails
        """
        client = self._connect_stac()

        print(f"Searching GFM catalog for date range: {start_date} to {end_date}")
        print(f"Bounding box: {bbox}")

        try:
            search_params = {
                "collections": [self.COLLECTION_ID],
                "bbox": bbox,
                "datetime": f"{start_date}/{end_date}"
            }

            if max_items is not None:
                search_params["max_items"] = max_items

            search = client.search(**search_params)
            items = search.item_collection()

            print(f"Found {len(items)} GFM items")

            # Print summary of dates found
            if len(items) > 0:
                dates = set([str(item.datetime)[:10] for item in items])
                print(f"Unique dates: {sorted(dates)}")

            return items

        except Exception as e:
            raise RuntimeError(f"Error searching STAC catalog: {e}")

    def create_daily_composites(
        self,
        items: ItemCollection,
        bbox: Tuple[float, float, float, float],
        target_dates: List[str]
    ) -> Dict[str, xr.DataArray]:
        """
        Create daily flood composites from STAC items using stackstac.

        Parameters
        ----------
        items : ItemCollection
            STAC ItemCollection with search results
        bbox : tuple
            Bounding box to clip data (minx, miny, maxx, maxy)
        target_dates : list
            List of dates (YYYY-MM-DD) to create composites for

        Returns
        -------
        dict
            Dictionary mapping date strings to xarray DataArrays
        """
        print("=" * 80)
        print("Creating Daily Composites")
        print("=" * 80)

        # Stack all items using stackstac
        print("Stacking STAC items...")
        stack = stackstac.stack(items, epsg=4326)

        # Extract flood extent band
        print("Extracting flood extent band...")
        stack_flood = stack.sel(band="ensemble_flood_extent")

        # Clip to bounding box
        print(f"Clipping to bbox: {bbox}")
        stack_flood_clipped = stack_flood.sel(
            x=slice(bbox[0], bbox[2]),
            y=slice(bbox[3], bbox[1])  # y is reversed
        )

        # Group by day and take maximum (merge overlapping tiles)
        print("Grouping by date and merging overlapping tiles...")
        stack_flood_daily = stack_flood_clipped.groupby("time.date").max()

        # Rename dimension back to 'time' and convert to datetime
        stack_flood_daily = stack_flood_daily.rename({"date": "time"})
        stack_flood_daily["time"] = stack_flood_daily.time.astype("datetime64[ns]")

        # Create dictionary of composites for target dates
        composites = {}
        for date_str in target_dates:
            try:
                # Select the specific date
                date_dt = datetime.strptime(date_str, "%Y-%m-%d")
                composite = stack_flood_daily.sel(time=date_str)

                print(f"Computing composite for {date_str}...")
                composite_computed = composite.compute()

                # Convert NaN to nodata value (255) and cast to uint8
                import numpy as np
                data_clean = np.where(
                    np.isnan(composite_computed.values),
                    255,
                    composite_computed.values
                ).astype(np.uint8)

                # Create new DataArray with cleaned data
                composite_clean = composite_computed.copy(data=data_clean)
                composite_clean = composite_clean.rio.write_nodata(255)

                composites[date_str] = composite_clean

                # Report statistics
                flood_pixels = (data_clean == 1).sum()
                valid_pixels = (data_clean != 255).sum()
                print(f"  ✅ {date_str}: {composite_clean.shape} pixels")
                print(f"     Flood pixels: {flood_pixels:,}, Valid pixels: {valid_pixels:,}")

            except KeyError:
                print(f"  ⚠️  No data available for {date_str}")
            except Exception as e:
                print(f"  ❌ Error processing {date_str}: {e}")

        print(f"\n✅ Created {len(composites)} daily composites")
        return composites

    def create_flood_point_shapefiles(
        self,
        composites: Dict[str, xr.DataArray],
        iso3_code: str,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Create point shapefiles from flood pixels (value = 1).

        Parameters
        ----------
        composites : dict
            Dictionary mapping date strings to xarray DataArrays
        iso3_code : str
            ISO3 country code for output naming
        output_dir : Path, optional
            Output directory (default: data/melissa/)

        Returns
        -------
        list
            List of paths to created shapefile directories
        """
        print("=" * 80)
        print("Creating Flood Point Shapefiles")
        print("=" * 80)

        if output_dir is None:
            output_dir = Path("data/melissa")

        # Create output directory
        shp_dir = output_dir / iso3_code / "shapefiles"
        shp_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []
        for date_str, composite in composites.items():
            try:
                print(f"\nCreating points for {date_str}...")

                # Get flood pixels (value == 1)
                flood_mask = composite.values == 1

                if not flood_mask.any():
                    print(f"  ⚠️  No flood pixels found for {date_str}")
                    continue

                # Get coordinates of flood pixels
                rows, cols = np.where(flood_mask)

                # Convert pixel indices to geographic coordinates
                # Using the xarray coordinates
                x_coords = composite.x.values[cols]
                y_coords = composite.y.values[rows]

                # Create points
                points = [Point(x, y) for x, y in zip(x_coords, y_coords)]

                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    {
                        'flood': np.ones(len(points), dtype=int),
                        'date': [date_str] * len(points),
                        'iso3': [iso3_code] * len(points)
                    },
                    geometry=points,
                    crs=composite.rio.crs
                )

                # Write shapefile
                shp_filename = f"gfm_flood_points_{iso3_code.lower()}_{date_str}.shp"
                shp_path = shp_dir / shp_filename

                gdf.to_file(shp_path)
                output_paths.append(shp_path)

                print(f"  ✅ Created {len(points):,} flood points")
                print(f"     Saved to: {shp_path}")

            except Exception as e:
                print(f"  ❌ Error creating shapefile for {date_str}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Successfully created {len(output_paths)} shapefiles")
        return output_paths

    def write_composites_to_cog(
        self,
        composites: Dict[str, xr.DataArray],
        iso3_code: str,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Write daily composites to Cloud-Optimized GeoTIFF files.

        Parameters
        ----------
        composites : dict
            Dictionary mapping date strings to xarray DataArrays
        iso3_code : str
            ISO3 country code for output naming
        output_dir : Path, optional
            Output directory (default: data/melissa/)

        Returns
        -------
        list
            List of paths to created COG files
        """
        print("=" * 80)
        print("Writing Composites to COG Files")
        print("=" * 80)

        if output_dir is None:
            output_dir = Path("data/melissa")

        # Initialize COG writer
        writer = COGWriter(output_base_dir=output_dir)

        output_paths = []
        for date_str, composite in composites.items():
            try:
                print(f"\nWriting {date_str}...")
                output_path = writer.write_flood_composite(
                    data_array=composite,
                    iso3=iso3_code,
                    date=date_str,
                    nodata_value=255,
                    overviews=True,
                    compression="DEFLATE"
                )
                output_paths.append(output_path)
                print(f"  ✅ Written to: {output_path}")

            except Exception as e:
                print(f"  ❌ Error writing {date_str}: {e}")

        print(f"\n✅ Successfully wrote {len(output_paths)} COG files")
        return output_paths

    def create_provenance_map(
        self,
        composites: Dict[str, xr.DataArray],
        iso3_code: str,
        output_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Create a provenance map showing which date each flood pixel came from.

        Layers composites from oldest to newest, assigning each date a unique value.
        The result shows the most recent observation date for each flooded pixel.

        Parameters
        ----------
        composites : dict
            Dictionary mapping date strings to xarray DataArrays
        iso3_code : str
            ISO3 country code for output naming
        output_dir : Path, optional
            Output directory (default: data/melissa/)

        Returns
        -------
        Path or None
            Path to the created provenance COG, or None if no flood pixels found
        """
        print("=" * 80)
        print("Creating Flood Provenance Map")
        print("=" * 80)

        if len(composites) == 0:
            print("No composites to process")
            return None

        if output_dir is None:
            output_dir = Path("data/melissa")

        # Sort dates (oldest to newest)
        sorted_dates = sorted(composites.keys())
        print(f"\nDates (oldest → newest): {sorted_dates}")

        # Get the spatial template from first composite
        first_composite = composites[sorted_dates[0]]

        # Initialize provenance array with nodata (0 = no observation)
        provenance = np.zeros(first_composite.shape, dtype=np.uint8)

        # Assign each date a unique value (1, 2, 3, ...)
        date_to_value = {date: idx + 1 for idx, date in enumerate(sorted_dates)}

        print("\nDate → Value mapping:")
        for date, value in date_to_value.items():
            print(f"  {date} → {value}")

        # Layer composites from oldest to newest
        total_flood_pixels = 0
        for date_str in sorted_dates:
            composite = composites[date_str]

            # Get flood pixels (value == 1)
            flood_mask = composite.values == 1
            flood_count = flood_mask.sum()

            if flood_count > 0:
                # Assign this date's value to flood pixels
                provenance[flood_mask] = date_to_value[date_str]
                total_flood_pixels += flood_count
                print(f"\n  {date_str}: Added {flood_count:,} flood pixels (value={date_to_value[date_str]})")

        if total_flood_pixels == 0:
            print("\n⚠️  No flood pixels found across all dates")
            return None

        print(f"\n✅ Total unique flood pixels in provenance map: {(provenance > 0).sum():,}")

        # Create xarray DataArray with same spatial reference
        provenance_da = first_composite.copy(data=provenance)
        provenance_da = provenance_da.rio.write_nodata(0)  # 0 = no flood observation

        # Add metadata
        provenance_da.attrs['long_name'] = 'Flood Observation Provenance (Date Index)'
        provenance_da.attrs['date_mapping'] = str(date_to_value)

        # Write to COG
        print("\nWriting provenance map to COG...")
        cog_dir = output_dir / iso3_code
        cog_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with date range
        start_date = sorted_dates[0]
        end_date = sorted_dates[-1]
        filename = f"gfm_flood_provenance_{iso3_code.lower()}_{start_date}_to_{end_date}.tif"
        output_path = cog_dir / filename

        writer = COGWriter(output_base_dir=output_dir)

        # Use the write method but we need to customize it
        # For now, write directly with rioxarray
        provenance_da.rio.to_raster(
            output_path,
            driver="GTiff",
            dtype="uint8",
            tiled=True,
            compress="DEFLATE",
            blockxsize=512,
            blockysize=512,
        )

        # Add overviews
        import rasterio
        from rasterio.enums import Resampling

        with rasterio.open(output_path, "r+") as dst:
            overview_levels = [2, 4, 8, 16]
            dst.build_overviews(overview_levels, Resampling.nearest)
            dst.update_tags(ns="rio_overview", resampling="nearest")

            # Add date mapping as metadata
            dst.update_tags(**{f"date_{val}": date for date, val in date_to_value.items()})

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Provenance map written to: {output_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Values: 0 (no flood), 1-{len(sorted_dates)} (date indices)")

        return output_path

    def process_latest(
        self,
        iso3_code: str,
        query_days: int = 12,
        num_composites: int = 3,
        output_dir: Optional[Path] = None,
        create_shapefiles: bool = True,
        create_provenance: bool = True,
        max_items: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Complete pipeline: query STAC, create composites, write COGs, shapefiles, and provenance map.

        This is the main method that:
        1. Loads the country boundary from FieldMaps.io
        2. Queries the last N days from STAC
        3. Identifies the most recent M dates with actual data
        4. Creates daily composites by merging overlapping tiles
        5. Writes composites as Cloud-Optimized GeoTIFFs
        6. Creates point shapefiles for flood pixels
        7. Creates provenance map showing most recent observation date

        Parameters
        ----------
        iso3_code : str
            ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
        query_days : int, optional
            Number of days to query from STAC (default: 12)
        num_composites : int, optional
            Number of most recent dates to create composites for (default: 3)
        output_dir : Path, optional
            Output directory for files (default: data/melissa/)
        create_shapefiles : bool, optional
            Whether to create point shapefiles for flood pixels (default: True)
        create_provenance : bool, optional
            Whether to create provenance map (default: True)
        max_items : int, optional
            Maximum number of STAC items to return. If None, returns all items.

        Returns
        -------
        dict
            Dictionary with keys: 'cogs', 'shapefiles', 'provenance'

        Raises
        ------
        ValueError
            If ISO3 code is invalid or no data found for country
        RuntimeError
            If STAC API connection or search fails

        Example
        -------
        >>> downloader = LatestGFMDownloader()
        >>> results = downloader.process_latest("JAM", query_days=12, num_composites=3)
        >>> print(f"Created {len(results['cogs'])} COG files")
        >>> print(f"Created {len(results['shapefiles'])} shapefiles")
        >>> print(f"Provenance map: {results['provenance']}")
        """
        print("=" * 80)
        print(f"Processing Latest GFM Data for {iso3_code}")
        print("=" * 80)

        # 1. Load country boundary
        gdf = self.load_country_boundary(iso3_code)

        # 2. Extract bounding box
        bbox = self.get_bbox_from_gdf(gdf)

        # 3. Query last N days from STAC
        start_date, end_date = self.get_latest_date_range(query_days)
        items = self.search_gfm_data(bbox, start_date, end_date, max_items)

        if len(items) == 0:
            print("⚠️  No items found in date range")
            return {'cogs': [], 'shapefiles': [], 'provenance': None}

        # 4. Get most recent M dates with actual data
        target_dates = self.get_recent_dates_from_items(items, num_composites)

        if len(target_dates) == 0:
            print("⚠️  No valid dates found")
            return {'cogs': [], 'shapefiles': [], 'provenance': None}

        # 5. Create daily composites
        composites = self.create_daily_composites(items, bbox, target_dates)

        if len(composites) == 0:
            print("⚠️  No composites created")
            return {'cogs': [], 'shapefiles': [], 'provenance': None}

        # 6. Write to COG files
        cog_paths = self.write_composites_to_cog(composites, iso3_code, output_dir)

        # 7. Create point shapefiles
        shp_paths = []
        if create_shapefiles:
            shp_paths = self.create_flood_point_shapefiles(composites, iso3_code, output_dir)

        # 8. Create provenance map
        provenance_path = None
        if create_provenance:
            provenance_path = self.create_provenance_map(composites, iso3_code, output_dir)

        print("=" * 80)
        print(f"✅ Pipeline Complete")
        print(f"   COG files: {len(cog_paths)}")
        print(f"   Shapefiles: {len(shp_paths)}")
        print(f"   Provenance map: {'✓' if provenance_path else '✗'}")
        print("=" * 80)

        return {'cogs': cog_paths, 'shapefiles': shp_paths, 'provenance': provenance_path}

    def download_latest(
        self,
        iso3_code: str,
        days: int = 12,
        max_items: Optional[int] = None
    ) -> ItemCollection:
        """
        Download latest GFM data for a country (legacy method).

        NOTE: This method is deprecated. Use process_latest() for the complete
        pipeline including compositing and COG writing.

        This method:
        1. Loads the country boundary from FieldMaps.io
        2. Extracts the bounding box
        3. Calculates the latest N days date range
        4. Queries the GFM STAC API
        5. Returns the ItemCollection

        Parameters
        ----------
        iso3_code : str
            ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
        days : int, optional
            Number of days to look back from today (default: 12)
        max_items : int, optional
            Maximum number of items to return. If None, returns all items.

        Returns
        -------
        ItemCollection
            STAC ItemCollection with GFM flood data

        Raises
        ------
        ValueError
            If ISO3 code is invalid or no data found for country
        RuntimeError
            If STAC API connection or search fails

        Example
        -------
        >>> downloader = LatestGFMDownloader()
        >>> items = downloader.download_latest("JAM")
        >>> print(f"Found {len(items)} items for Jamaica")
        """
        print("=" * 80)
        print(f"Downloading Latest GFM Data for {iso3_code}")
        print("=" * 80)

        # 1. Load country boundary
        gdf = self.load_country_boundary(iso3_code)

        # 2. Extract bounding box
        bbox = self.get_bbox_from_gdf(gdf)

        # 3. Get latest date range
        start_date, end_date = self.get_latest_date_range(days)

        # 4. Search STAC catalog
        items = self.search_gfm_data(bbox, start_date, end_date, max_items)

        print("=" * 80)
        print(f"✅ Successfully retrieved {len(items)} items")
        print("=" * 80)

        return items


def process_latest_gfm(
    iso3_code: str,
    query_days: int = 12,
    num_composites: int = 3,
    output_dir: Optional[Path] = None,
    create_shapefiles: bool = True,
    create_provenance: bool = True,
    max_items: Optional[int] = None,
    stac_url: Optional[str] = None
) -> Dict[str, any]:
    """
    Complete pipeline to process latest GFM data for a country.

    This convenience function:
    1. Queries the last N days from STAC
    2. Identifies the most recent M dates with actual data
    3. Creates daily composites by merging overlapping tiles
    4. Writes composites as Cloud-Optimized GeoTIFFs
    5. Creates point shapefiles for flood pixels
    6. Creates provenance map showing most recent observation date

    Parameters
    ----------
    iso3_code : str
        ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
    query_days : int, optional
        Number of days to query from STAC (default: 12)
    num_composites : int, optional
        Number of most recent dates to create composites for (default: 3)
    output_dir : Path, optional
        Output directory for files (default: data/melissa/)
    create_shapefiles : bool, optional
        Whether to create point shapefiles for flood pixels (default: True)
    create_provenance : bool, optional
        Whether to create provenance map (default: True)
    max_items : int, optional
        Maximum number of STAC items to return. If None, returns all items.
    stac_url : str, optional
        STAC API endpoint URL. Defaults to EODC STAC endpoint.

    Returns
    -------
    dict
        Dictionary with keys: 'cogs', 'shapefiles', 'provenance'

    Example
    -------
    >>> from ds_flood_gfm.download_latest import process_latest_gfm
    >>> results = process_latest_gfm("JAM", query_days=12, num_composites=3)
    >>> print(f"Created {len(results['cogs'])} COG files")
    >>> print(f"Created {len(results['shapefiles'])} shapefiles")
    >>> print(f"Provenance map: {results['provenance']}")
    """
    downloader = LatestGFMDownloader(stac_url=stac_url)
    return downloader.process_latest(
        iso3_code=iso3_code,
        query_days=query_days,
        num_composites=num_composites,
        output_dir=output_dir,
        create_shapefiles=create_shapefiles,
        create_provenance=create_provenance,
        max_items=max_items
    )


def download_latest_gfm(
    iso3_code: str,
    days: int = 12,
    max_items: Optional[int] = None,
    stac_url: Optional[str] = None
) -> ItemCollection:
    """
    Convenience function to download latest GFM data for a country (legacy).

    NOTE: This function is deprecated. Use process_latest_gfm() for the complete
    pipeline including compositing and COG writing.

    This is a simple wrapper around LatestGFMDownloader for quick access.

    Parameters
    ----------
    iso3_code : str
        ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
    days : int, optional
        Number of days to look back from today (default: 12)
    max_items : int, optional
        Maximum number of items to return. If None, returns all items.
    stac_url : str, optional
        STAC API endpoint URL. Defaults to EODC STAC endpoint.

    Returns
    -------
    ItemCollection
        STAC ItemCollection with GFM flood data

    Example
    -------
    >>> from ds_flood_gfm.download_latest import download_latest_gfm
    >>> items = download_latest_gfm("JAM")  # Jamaica
    >>> print(f"Found {len(items)} items")
    >>>
    >>> # Access item properties
    >>> for item in items:
    >>>     print(f"Date: {item.datetime}, ID: {item.id}")
    >>>     print(f"Assets: {list(item.assets.keys())}")
    """
    downloader = LatestGFMDownloader(stac_url=stac_url)
    return downloader.download_latest(iso3_code, days, max_items)


def main():
    """
    Example usage demonstrating the complete pipeline.

    Tests the pipeline with a sample country to create COG composites and shapefiles.
    """
    print("\n" + "=" * 80)
    print("GFM LATEST DATA PIPELINE - EXAMPLE")
    print("=" * 80)

    # Example: Complete pipeline (recommended)
    print("\n" + "=" * 80)
    print("Complete Pipeline (Query + Composite + COG + Shapefiles)")
    print("=" * 80)

    test_iso3 = "JAM"  # Jamaica as example
    test_name = "Jamaica"

    try:
        print(f"\nProcessing latest GFM data for {test_name}...")

        # Run the complete pipeline
        results = process_latest_gfm(
            iso3_code=test_iso3,
            query_days=12,  # Query last 12 days
            num_composites=3,  # Create composites for 3 most recent dates
            output_dir=Path("data/melissa"),  # Output directory
            create_shapefiles=True  # Create point shapefiles
        )

        print(f"\n✅ Success! Pipeline completed for {test_name}")
        print(f"   COG files: {len(results['cogs'])}")
        print(f"   Shapefiles: {len(results['shapefiles'])}")
        print(f"   Provenance map: {'✓' if results['provenance'] else '✗'}")

        if len(results['cogs']) > 0:
            print("\nCOG files:")
            for path in results['cogs']:
                print(f"  - {path}")

        if len(results['shapefiles']) > 0:
            print("\nShapefiles:")
            for path in results['shapefiles']:
                print(f"  - {path}")

        if results['provenance']:
            print(f"\nProvenance map:")
            print(f"  - {results['provenance']}")

    except Exception as e:
        print(f"\n❌ Error processing {test_name}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
