import geopandas as gpd
from fsspec.implementations.http import HTTPFileSystem
from shapely import wkb
import pandas as pd


def load_adm0_lowres():
    """Load low-resolution admin0 countries data from Natural Earth.

    Returns:
        gpd.GeoDataFrame: World countries with standardized column names
    """
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)

    rename_mapper = {"POP_EST": "pop_est",
                     "CONTINENT": "continent",
                     "ADMIN": "name",
                     "ADM0_A3": "iso_a3",
                     "GDP_MD": "gdp_md_est"}
    world.rename(rename_mapper, axis="columns", inplace=True)

    return world


def load_adm0_fieldmaps(iso3_code):
    """Load admin0 country boundary from FieldMaps.io humanitarian data.

    This uses the edge-matched humanitarian boundaries from FieldMaps.io,
    which are based on UN Common Operational Datasets (COD) when available,
    falling back to geoBoundaries.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code (e.g., 'NGA', 'ETH', 'AFG')

    Returns:
        gpd.GeoDataFrame: Country boundary with columns [iso_3, iso_2, adm0_name, adm0_id, geometry]

    Example:
        >>> gdf_nga = load_adm0_fieldmaps('NGA')
        >>> print(gdf_nga.adm0_name.iloc[0])
        Nigeria
    """
    GLOBAL_ADM1 = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"

    filesystem = HTTPFileSystem()
    filters = [("iso_3", "=", iso3_code)]

    # Read ADM1 data for the country
    gdf_adm1 = gpd.read_parquet(GLOBAL_ADM1, filesystem=filesystem, filters=filters)

    if len(gdf_adm1) == 0:
        raise ValueError(f"No data found for ISO3 code: {iso3_code}")

    # Dissolve to ADM0 by grouping all ADM1 units
    gdf_adm0 = gdf_adm1.dissolve(by="iso_3", as_index=False)

    # Keep relevant ADM0 columns
    cols_to_keep = ["iso_3", "iso_2", "adm0_name", "adm0_id", "geometry"]
    gdf_adm0 = gdf_adm0[cols_to_keep]

    return gdf_adm0


def load_admin_from_blob(iso3_code, adm_level, stage="dev"):
    """Load administrative boundaries from Azure Blob Storage (FAST!).

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
        adm_level: Administrative level (1, 2, or 3)
        stage: Azure stage - "dev" or "prod" (default: "dev")

    Returns:
        gpd.GeoDataFrame: Administrative boundaries

    Raises:
        ValueError: If data not found in blob storage
    """
    from ocha_stratus import load_parquet_from_blob
    from ds_flood_gfm.constants import PROJECT_PREFIX

    iso3_lower = iso3_code.lower()
    blob_name = f"{PROJECT_PREFIX}/raw/codab/{iso3_lower}/{iso3_lower}_adm{adm_level}.parquet"

    try:
        # Load parquet from blob (matches test_codab_blob_read.py)
        df = load_parquet_from_blob(
            blob_name=blob_name,
            stage=stage,
            container_name="projects"
        )

        # Convert WKB geometry back to shapely geometries
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(wkb.loads)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        return gdf

    except Exception as e:
        raise ValueError(f"Could not load from blob: {blob_name}. Error: {e}")


def load_fieldmaps_parquet(iso3_code, adm_level=1, admin_source="http"):
    """Load administrative boundaries from blob storage or FieldMaps.io.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'CUB', 'HTI')
        adm_level: Administrative level (0, 1, 2, 3, or 4). Defaults to 1.
        admin_source: Source to load from - "blob" for Azure blob storage (fast!),
                     "http" for FieldMaps.io HTTP (default: "http")

    Returns:
        gpd.GeoDataFrame: Administrative boundaries filtered by ISO3 code

    Raises:
        ValueError: If adm_level is not in range 0-4
        ValueError: If no data found for the ISO3 code at that level

    Example:
        >>> # Load Jamaica's admin1 divisions from HTTP (default)
        >>> gdf_parishes = load_fieldmaps_parquet('JAM', adm_level=1)
        >>> print(f"Loaded {len(gdf_parishes)} admin1 divisions")

        >>> # Load from blob storage (fast!)
        >>> gdf = load_fieldmaps_parquet('JAM', adm_level=3, admin_source="blob")
    """
    # Load from blob storage if requested
    if admin_source == "blob":
        return load_admin_from_blob(iso3_code, adm_level)

    # Fallback to HTTP (slower but works for all countries)
    base_url = "https://data.fieldmaps.io/edge-matched/humanitarian/intl"

    if adm_level == 0:
        url = f"{base_url}/adm0_polygons.parquet"
    elif adm_level == 1:
        url = f"{base_url}/adm1_polygons.parquet"
    elif adm_level == 2:
        url = f"{base_url}/adm2_polygons.parquet"
    elif adm_level == 3:
        url = f"{base_url}/adm3_polygons.parquet"
    elif adm_level == 4:
        url = f"{base_url}/adm4_polygons.parquet"
    else:
        raise ValueError(f"adm_level must be 0-4, got: {adm_level}")

    # Load data with ISO3 filter
    filesystem = HTTPFileSystem()
    filters = [("iso_3", "=", iso3_code)]

    try:
        gdf = gpd.read_parquet(url, filesystem=filesystem, filters=filters)
    except Exception as e:
        raise ValueError(f"Could not load data for ISO3 code: {iso3_code} at admin level {adm_level}. Error: {e}")

    if len(gdf) == 0:
        raise ValueError(f"No data found for ISO3 code: {iso3_code} at admin level {adm_level}")

    return gdf


def get_highest_admin_level(iso3_code, max_level=4):
    """Find the highest available administrative level for a country.

    Checks from highest (adm4) down to lowest (adm1) to find the most granular
    administrative boundaries available for the country.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'NGA', 'ETH')
        max_level: Maximum admin level to check (default: 4)

    Returns:
        tuple: (admin_level, GeoDataFrame) with the highest available admin level
               and its corresponding data

    Raises:
        ValueError: If no admin data available for the country (checked 1-4)

    Example:
        >>> # Find highest admin level for Jamaica
        >>> level, gdf = get_highest_admin_level('JAM')
        >>> print(f"Highest level: ADM{level} with {len(gdf)} divisions")
        Highest level: ADM2 with 133 divisions
    """
    for adm_level in range(max_level, 0, -1):  # Check 4, 3, 2, 1
        try:
            gdf = load_fieldmaps_parquet(iso3_code, adm_level=adm_level)
            if len(gdf) > 0:
                return adm_level, gdf
        except (ValueError, Exception):
            continue

    raise ValueError(f"No administrative boundary data found for ISO3 code: {iso3_code}")


def calculate_admin_population(flood_points_gdf, admin_gdf, adm_level):
    """Calculate affected population by administrative division.

    Performs a spatial join between flood points (with population values) and
    administrative boundaries, then sums population by admin division.

    Args:
        flood_points_gdf: GeoDataFrame with flood point locations and 'population' column
        admin_gdf: GeoDataFrame with administrative boundaries
        adm_level: Administrative level (1, 2, 3, or 4) for column name selection

    Returns:
        gpd.GeoDataFrame: Admin boundaries with added columns:
            - affected_pop: Total affected population in each division
            - flood_pixels: Number of flood pixels in each division

    Example:
        >>> # Create flood points GeoDataFrame
        >>> gdf_floods = gpd.GeoDataFrame(flood_points, crs='EPSG:4326')
        >>> # Load admin boundaries
        >>> adm_level, gdf_admin = get_highest_admin_level('JAM')
        >>> # Calculate affected population
        >>> gdf_choropleth = calculate_admin_population(gdf_floods, gdf_admin, adm_level)
        >>> print(gdf_choropleth[['adm2_name', 'affected_pop', 'flood_pixels']])
    """
    # Ensure both have same CRS
    if flood_points_gdf.crs != admin_gdf.crs:
        flood_points_gdf = flood_points_gdf.to_crs(admin_gdf.crs)

    # Spatial join: assign each flood point to an admin division
    joined = gpd.sjoin(flood_points_gdf, admin_gdf, how='left', predicate='within')

    # Column name for admin ID depends on level
    adm_id_col = f"adm{adm_level}_id"
    adm_name_col = f"adm{adm_level}_name"

    # Group by admin division and sum population
    pop_by_admin = joined.groupby(adm_id_col).agg({
        'population': 'sum',
        'geometry': 'count'  # Count flood pixels
    }).reset_index()

    pop_by_admin.columns = [adm_id_col, 'affected_pop', 'flood_pixels']

    # Merge back to admin boundaries
    result = admin_gdf.merge(pop_by_admin, on=adm_id_col, how='left')

    # Fill NaN with 0 (divisions with no floods)
    result['affected_pop'] = result['affected_pop'].fillna(0)
    result['flood_pixels'] = result['flood_pixels'].fillna(0)

    return result