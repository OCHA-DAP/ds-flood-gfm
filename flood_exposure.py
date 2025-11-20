import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.center(mo.md("# GFM Flood Exposure Mapping"))
    return


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    import ocha_stratus as stratus
    from dotenv import load_dotenv, find_dotenv
    import rioxarray as rxr
    from rasterio.features import geometry_mask
    import exactextract
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import plotly.express as px
    import re

    GHSL_RASTER_BLOB_PATH_3s = "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif"
    GHSL_RASTER_BLOB_PATH_30s = "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"
    # JRC_FLOOD_ZIP = "ds-flood-gfm/processed/polygon/CUB_20251102_20251103_20251104_20251105_nopop_cumulative.shp.zip"
    # JRC_FLOOD_SHP = "data/data.shp"

    _ = load_dotenv(find_dotenv(usecwd=True))
    return (
        GHSL_RASTER_BLOB_PATH_3s,
        exactextract,
        geometry_mask,
        mcolors,
        np,
        plt,
        px,
        re,
        stratus,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Select data to plot
    """)
    return


@app.cell
def _(mo):
    iso3_dropdown = mo.ui.dropdown(
        label="Select a country", options=["CUB", "HTI", "JAM", "PHL"], value="HTI"
    )
    adm_dropdown = mo.ui.dropdown(
        label="Select an admin level", options=[0, 1, 2, 3], value=3
    )

    mo.hstack([iso3_dropdown, adm_dropdown], justify="start")
    return adm_dropdown, iso3_dropdown


@app.cell
def _(iso3_dropdown, stratus):
    cnt = stratus.get_container_client("projects")
    blob_list = cnt.list_blobs(
        name_starts_with=f"ds-flood-gfm/processed/polygon/{iso3_dropdown.value}"
    )
    blob_names = [blob.name for blob in blob_list]
    return (blob_names,)


@app.cell
def _(blob_names, mo):
    shp_dropdown = mo.ui.dropdown(
        label="Select a shapefile", options=blob_names, value=blob_names[0]
    )
    shp_dropdown
    return (shp_dropdown,)


@app.cell
def _(mo, stratus):
    @mo.persistent_cache
    def get_adm(iso3, adm_level):
        return stratus.codab.load_codab_from_fieldmaps(iso3, adm_level)

    @mo.persistent_cache
    def get_pop(blob_path):
        return stratus.open_blob_cog(blob_path, container_name="raster").squeeze(
            drop=True
        )

    @mo.persistent_cache
    def get_flood(flood_zip, flood_shp):
        return stratus.load_shp_from_blob(flood_zip, flood_shp)
    return get_adm, get_flood, get_pop


@app.cell
def _(
    GHSL_RASTER_BLOB_PATH_3s,
    adm_dropdown,
    get_adm,
    get_flood,
    get_pop,
    iso3_dropdown,
    shp_dropdown,
):
    target_crs = "EPSG:32618"  # UTM Zone 17N

    # Get the data
    gdf_adm = get_adm(iso3_dropdown.value, adm_dropdown.value)
    da_pop = get_pop(GHSL_RASTER_BLOB_PATH_3s)
    gdf_flood = get_flood(shp_dropdown.value, "data/data.shp")
    return da_pop, gdf_adm, gdf_flood, target_crs


@app.cell
def _(da_pop, gdf_adm):
    minx, miny, maxx, maxy = gdf_adm.total_bounds
    da_clip = da_pop.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    return (da_clip,)


@app.cell
def _(mo):
    buffer_dropdown = mo.ui.number(
        label="Select flood buffer distance (m)", start=0, stop=1000, value=100
    )
    buffer_dropdown
    return (buffer_dropdown,)


@app.cell
def _(mo):
    map_display = mo.ui.radio(
        label="Select data to display on map",
        options=[
            "flood exposure",
            "flood polygons (original)",
            "flood polygons (buffered)",
        ],
        value="flood exposure",
        inline=True,
    )
    map_display
    return (map_display,)


@app.cell
def _(buffer_dropdown, da_clip, gdf_adm, gdf_flood, target_crs):
    # Set everything to a local CRS
    da_clip_r = da_clip.rio.reproject(target_crs)
    gdf_adm_r = gdf_adm.to_crs(target_crs)
    gdf_flood_r = gdf_flood.to_crs(target_crs)

    # Now clip and buffer the flood polygons
    bbox = gdf_adm_r.total_bounds  # [minx, miny, maxx, maxy]
    gdf_flood_clipped = gdf_flood_r.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    gdf_flood_buffered = gdf_flood_clipped.copy()
    gdf_flood_buffered.geometry = gdf_flood_buffered.buffer(buffer_dropdown.value)
    return da_clip_r, gdf_adm_r, gdf_flood_buffered, gdf_flood_clipped


@app.cell
def _(da_clip_r, gdf_flood_buffered, geometry_mask):
    # Create a mask for flooded areas on the population grid
    # This is the cell that will take the longest
    flood_mask = geometry_mask(
        gdf_flood_buffered.geometry,
        transform=da_clip_r.rio.transform(),
        invert=True,
        out_shape=da_clip_r.shape,
    )

    # Apply mask to population data - only keep population in flooded areas
    da_pop_flood = da_clip_r.where(flood_mask, 0)
    return (da_pop_flood,)


@app.cell
def _(adm_dropdown, da_pop_flood, exactextract, gdf_adm_r, np):
    # Use exactextract for zonal statistics
    df_stats = exactextract.exact_extract(
        da_pop_flood,
        gdf_adm_r,
        "sum",
        include_cols=[f"adm{adm_dropdown.value}_id"],
        output="pandas",
    )

    gdf_result = gdf_adm_r.merge(df_stats)
    gdf_result["pop_exposed"] = np.ceil(gdf_result["sum"])
    gdf_result[f"adm{adm_dropdown.value}_name"] = gdf_result[
        f"adm{adm_dropdown.value}_name"
    ].fillna(gdf_result[f"adm{adm_dropdown.value}_id"])
    return (gdf_result,)


@app.cell
def _(re):
    def parse_flood_string(s):
        dates = re.findall(r"(\d{4})(\d{2})(\d{2})", s)
        formatted_dates = [f"{year}-{month}-{day}" for year, month, day in dates]
        # Extract the descriptive word (assumes it's before .shp.zip)
        match = re.search(r"_([a-zA-Z]+)\.shp\.zip$", s)
        descriptor = match.group(1) if match else "flood"
        descriptor = descriptor.capitalize()
        return f"{descriptor} flooding: {', '.join(formatted_dates)}"
    return (parse_flood_string,)


@app.cell
def _(
    adm_dropdown,
    buffer_dropdown,
    gdf_flood_buffered,
    gdf_flood_clipped,
    gdf_result,
    map_display,
    parse_flood_string,
    px,
    shp_dropdown,
):
    total_exposed = int(gdf_result["pop_exposed"].sum())

    if map_display.value == "flood exposure":
        gdf_result.geometry = gdf_result.geometry.simplify(tolerance=50)
        gdf_wgs84 = gdf_result.to_crs("EPSG:4326")
        bounds = gdf_wgs84.total_bounds
        # Calculate 90th percentile for color scale
        max_color_value = gdf_wgs84["pop_exposed"].quantile(0.99)

        fig = px.choropleth_map(
            gdf_wgs84,
            geojson=gdf_wgs84.geometry,
            locations=gdf_wgs84.index,
            color="pop_exposed",
            color_continuous_scale=[
                (0, "white"),
                (0.001, "white"),
                (0.001, "#fee5d9"),
                (1, "#a50f15"),
            ],
            range_color=[0, max_color_value],
            zoom=7,
            center={
                "lat": (bounds[1] + bounds[3]) / 2,
                "lon": (bounds[0] + bounds[2]) / 2,
            },
            hover_name=f"adm{adm_dropdown.value}_name",
            hover_data={"pop_exposed": ":,.0f"},
            height=600,
        )

        fig.update_traces(marker_line_color="lightgrey", marker_line_width=0.5)

        # Update legend title and format number in title with comma
        title = parse_flood_string(shp_dropdown.value)
        title_suffix = f" - ({buffer_dropdown.value}m buffer) - {total_exposed} people"
        fig.update_layout(title=title + title_suffix)
    else:
        gdf_plot = (
            gdf_flood_buffered.copy()
            if "buffered" in map_display.value
            else gdf_flood_clipped.copy()
        )
        gdf_plot.geometry = gdf_plot.geometry.simplify(tolerance=50)
        gdf_wgs84 = gdf_plot.to_crs("EPSG:4326")
        bounds = gdf_wgs84.total_bounds

        fig = px.choropleth_map(
            gdf_wgs84,
            geojson=gdf_wgs84.geometry,
            locations=gdf_wgs84.index,
            zoom=7,
            center={
                "lat": (bounds[1] + bounds[3]) / 2,
                "lon": (bounds[0] + bounds[2]) / 2,
            },
            height=600,
        )
        title = parse_flood_string(shp_dropdown.value)
        title_suffix = (
            f" - ({buffer_dropdown.value}m buffer)"
            if "buffered" in map_display.value
            else ""
        )
        fig.update_layout(
            title=title + title_suffix,
        )

    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Sanity check outputs
    """)
    return


@app.cell
def _(adm_dropdown, gdf_result, mo):
    adms = sorted(list(gdf_result[f"adm{adm_dropdown.value}_name"].unique()))
    check_dropdown = mo.ui.dropdown(
        label="Select an admin to check",
        options=adms,
        value=adms[0],
        searchable=True,
    )
    check_dropdown
    return (check_dropdown,)


@app.cell
def _(
    adm_dropdown,
    buffer_dropdown,
    check_dropdown,
    da_clip_r,
    gdf_flood_buffered,
    gdf_flood_clipped,
    gdf_result,
    mcolors,
    mo,
    np,
    plt,
):
    # Input parameters
    admin_name = check_dropdown.value
    buffer_distance = buffer_dropdown.value  # In meters, set to 0 for no buffer

    # Find and extract the admin region
    admin_region = gdf_result[gdf_result[f"adm{adm_dropdown.value}_name"] == admin_name]

    # Clip to admin boundary
    pop_clipped = da_clip_r.rio.clip(admin_region.geometry, drop=True)

    _bounds = admin_region.total_bounds
    flood_clipped = gdf_flood_clipped.cx[
        _bounds[0] - 1000 : _bounds[2] + 1000,
        _bounds[1] - 1000 : _bounds[3] + 1000,
    ]

    _fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap: white for zero, then Blues
    colors = ["white"] + [plt.cm.Blues(i) for i in np.linspace(0.3, 1.0, 256)]
    custom_cmap = mcolors.ListedColormap(colors)

    pop_clipped_plot = pop_clipped.where(pop_clipped > 0)  # Mask zeros
    pop_clipped_plot.plot(
        ax=ax,
        cmap=custom_cmap,
        alpha=0.7,
        add_colorbar=True,
        norm=mcolors.LogNorm(vmin=0.1, vmax=pop_clipped.max()),
    )

    admin_region.boundary.plot(
        ax=ax, color="black", linewidth=2, label="Admin Boundary"
    )

    if len(flood_clipped) > 0:
        flood_clipped.boundary.plot(
            ax=ax, color="red", linewidth=0.5, label="Flood Areas"
        )

        if buffer_distance > 0:
            gdf_flood_buffered.boundary.plot(
                ax=ax,
                color="orange",
                linewidth=0.5,
                linestyle="--",
                label=f"Flood Buffer ({buffer_distance}m)",
            )

    ax.set_title(f"Flood Exposure Check: {admin_name}")
    ax.legend()
    ax.set_axis_off()

    mo.accordion({"View plot": _fig})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Output overall exposure
    """)
    return


@app.cell
def _(mo):
    save_to_blob = mo.ui.switch(label="Save to blob", value=False)
    save_to_blob
    return (save_to_blob,)


@app.cell
def _(adm_dropdown, gdf_result):
    df_output = gdf_result[
        [
            f"adm{adm_dropdown.value}_name",
            f"adm{adm_dropdown.value}_src",
            "pop_exposed",
        ]
    ].sort_values("pop_exposed", ascending=False)
    return (df_output,)


@app.cell
def _(df_output):
    df_output
    return


@app.cell
def _(buffer_dropdown, df_output, save_to_blob, shp_dropdown, stratus):
    if save_to_blob.value:
        # This is a bit ugly...
        output_fname = f"{shp_dropdown.value.split('/')[-1].split('.')[0].replace('_nopop', '')}_b{buffer_dropdown.value}.csv"
        stratus.upload_csv_to_blob(
            df_output,
            blob_name=f"ds-flood-gfm/processed/exposed_population/{output_fname}",
            container_name="projects",
            stage="dev",
        )
    return


@app.cell
def _(mo):
    mo.md(r"""
    <!-- ## Merge with the existing flood points parquet files to compare methods.

    Will fail if the cache is not on your local machine! -->
    """)
    return


@app.cell
def _():
    # latest_cache_paths = {
    #     "CUB": "data/cache/CUB_20251026_20251027_20251029_20251030_ghsl_cumulative/flood_points.parquet",
    #     "JAM": "data/cache/JAM_20251024_20251027_20251029_20251030_ghsl_cumulative/flood_points.parquet",
    #     "HTI": "data/cache/HTI_20251026_20251029_20251030_20251031_ghsl_cumulative/flood_points.parquet"
    # }

    # gdf_points = gpd.read_parquet(latest_cache_paths[iso3_dropdown.value])
    # gdf_points = gdf_points.to_crs(target_crs)

    # count_col = "population_adjusted"

    # # Spatial join to assign admin region to each point
    # gdf_points_with_admin = gpd.sjoin(gdf_points, gdf_result, how='left', predicate='within')

    # # Group by admin region and sum population
    # admin_pop_sum = gdf_points_with_admin.groupby(f"adm{adm_dropdown.value}_id")[count_col].sum().reset_index()

    # gdf_result_summary = gdf_result.copy()
    # gdf_result_summary = gdf_result_summary.merge(admin_pop_sum, how='left').fillna(0)
    # gdf_result_summary = gdf_result_summary.rename(columns={"pop_exposed": "jrc_pop_exposed", count_col: "chd_gfm_pop_exposed"})

    # df_output = gdf_result_summary[[f"adm{adm_dropdown.value}_name", f"adm{adm_dropdown.value}_src", "jrc_pop_exposed", "chd_gfm_pop_exposed"]].sort_values("chd_gfm_pop_exposed", ascending=False)

    # fname = f"{iso3_dropdown.value}_adm{adm_dropdown.value}_pop_exposure.csv"
    # stratus.upload_csv_to_blob(
    #     df_output,
    #     blob_name=f"ds-flood-gfm/processed/{fname}",
    #     container_name="projects",
    #     stage="dev"
    # )
    return


if __name__ == "__main__":
    app.run()
