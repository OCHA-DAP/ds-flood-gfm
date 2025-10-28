"""Country-specific configuration for flood monitoring analysis.

This module provides configuration settings for different countries including:
- Bounding boxes for map extent
- Admin level preferences for choropleth maps
- Country names and ISO codes

Usage:
    from ds_flood_gfm.country_config import get_country_config

    config = get_country_config('JAM')
    bbox = config['bbox']
"""

# Country configurations
COUNTRY_CONFIGS = {
    'JAM': {
        'name': 'Jamaica',
        'iso3': 'JAM',
        'iso2': 'JM',
        'bbox': [-78.3689286, 17.7055633, -76.1829608, 18.5251491],  # [min_lon, min_lat, max_lon, max_lat]
        'choropleth_adm_level': 3,  # Preferred admin level for choropleth maps
        'population_raster': 'ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif',
    },
    'HTI': {
        'name': 'Haiti',
        'iso3': 'HTI',
        'iso2': 'HT',
        'bbox': [-74.5, 18.0, -71.6, 20.1],  # [min_lon, min_lat, max_lon, max_lat]
        'choropleth_adm_level': 3,
        'population_raster': 'ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif',
    },
    'CUB': {
        'name': 'Cuba',
        'iso3': 'CUB',
        'iso2': 'CU',
        'bbox': [-85.0, 19.8, -74.0, 23.3],  # [min_lon, min_lat, max_lon, max_lat]
        'choropleth_adm_level': 2,  # Cuba may have different admin structure
        'population_raster': 'ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif',
    },
}


def get_country_config(iso3_code):
    """Get configuration for a specific country by ISO3 code.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code (e.g., 'JAM', 'HTI', 'CUB')

    Returns:
        dict: Country configuration dictionary

    Raises:
        ValueError: If country code not found in configuration

    Example:
        >>> config = get_country_config('JAM')
        >>> print(config['name'])
        Jamaica
        >>> print(config['bbox'])
        [-78.3689286, 17.7055633, -76.1829608, 18.5251491]
    """
    iso3_upper = iso3_code.upper()

    if iso3_upper not in COUNTRY_CONFIGS:
        available = ', '.join(COUNTRY_CONFIGS.keys())
        raise ValueError(
            f"Country '{iso3_code}' not found in configuration. "
            f"Available countries: {available}"
        )

    return COUNTRY_CONFIGS[iso3_upper]


def get_bbox(iso3_code):
    """Get bounding box for a country.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code

    Returns:
        list: Bounding box as [min_lon, min_lat, max_lon, max_lat]

    Example:
        >>> bbox = get_bbox('JAM')
        >>> print(bbox)
        [-78.3689286, 17.7055633, -76.1829608, 18.5251491]
    """
    config = get_country_config(iso3_code)
    return config['bbox']


def get_choropleth_admin_level(iso3_code):
    """Get preferred admin level for choropleth maps.

    Args:
        iso3_code: ISO 3166-1 alpha-3 country code

    Returns:
        int: Admin level (1-4)

    Example:
        >>> level = get_choropleth_admin_level('JAM')
        >>> print(level)
        3
    """
    config = get_country_config(iso3_code)
    return config['choropleth_adm_level']


def list_countries():
    """List all available countries in configuration.

    Returns:
        list: List of dictionaries with country info

    Example:
        >>> countries = list_countries()
        >>> for c in countries:
        ...     print(f"{c['iso3']}: {c['name']}")
        JAM: Jamaica
        HTI: Haiti
        CUB: Cuba
    """
    return [
        {
            'iso3': iso3,
            'iso2': config['iso2'],
            'name': config['name']
        }
        for iso3, config in COUNTRY_CONFIGS.items()
    ]
