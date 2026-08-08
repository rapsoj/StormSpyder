import os
import shutil

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from PIL import Image
from shapely.geometry import Point
from sklearn.cluster import DBSCAN


def get_image_paths(storm_type):
    folder_path = os.path.join('temp', storm_type)
    all_items = os.listdir(folder_path)
    files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    return files


def pixel_to_geospatial(row, col, transform):
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def is_similar_color(color1, color2, tolerance):
    return np.linalg.norm(np.array(color1) - np.array(color2)) < tolerance


def cluster_points(geo_df):
    if len(geo_df) > 0:
        points = np.array(geo_df.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
        min_distance = 33392.607
        epsilon = 5 * min_distance
        dbscan = DBSCAN(eps=epsilon, min_samples=1)
        labels = dbscan.fit_predict(points)
        geo_df['cluster'] = labels
        cluster_counts = geo_df['cluster'].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= 4].index
        geo_df = geo_df[geo_df['cluster'].isin(valid_clusters)]
        return geo_df


def detect_strike_probability(storm_type, date):
    strike_probabilities = {
        0.05: (255, 0, 255, 170),
        0.15: (255, 78, 0, 120),
        0.25: (255, 176, 0, 90),
        0.35: (255, 255, 0, 70),
        0.45: (40, 255, 0, 150),
        0.55: (0, 140, 47, 50),
        0.65: (2, 255, 255, 100),
        0.75: (2, 127, 254, 50),
        0.85: (0, 0, 255, 50),
        0.95: (122, 16, 178, 20),
    }

    original_image = Image.open(os.path.join('temp', storm_type, f'{date}.png'))
    original_array = np.array(original_image)
    dataset = rasterio.open('ref_maps/georeferenced_map.tif')
    transform = dataset.transform

    geo_df_list = []
    for strike_prob in strike_probabilities.keys():
        target_color = strike_probabilities[strike_prob][:3]
        color_tolerance = strike_probabilities[strike_prob][3]

        matching_pixels = np.zeros((original_array.shape[0], original_array.shape[1]), dtype=bool)
        for row in range(original_array.shape[0]):
            for col in range(original_array.shape[1]):
                pixel_color = original_array[row, col, :3]
                if is_similar_color(pixel_color, target_color, color_tolerance):
                    matching_pixels[row, col] = True

        pixel_indices = np.argwhere(matching_pixels)
        geospatial_points = [pixel_to_geospatial(row, col, transform) for row, col in pixel_indices]
        geometry = [Point(x, y) for x, y in geospatial_points]
        geo_df_strike = gpd.GeoDataFrame(geometry=geometry, crs=dataset.crs)
        geo_df_strike['type'] = storm_type
        geo_df_strike['strike_probability'] = strike_prob
        geo_df_list.append(geo_df_strike)

    geo_df = gpd.GeoDataFrame(pd.concat(geo_df_list, ignore_index=True), crs=dataset.crs)
    geo_df = geo_df.sort_values(by=['geometry', 'strike_probability'], ascending=[True, False])
    geo_df = geo_df.drop_duplicates(subset='geometry', keep='first')
    geo_df = geo_df.reset_index(drop=True)
    geo_df = cluster_points(geo_df)
    return geo_df


def load_pop_map():
    file_path = 'ref_maps/gpw_v4_population_count_rev11_2020_2pt5_min.tif'
    with rasterio.open(file_path) as src:
        pop_map = src.read(1)
        nodata_value = src.nodata
        metadata = src.meta
        raster_crs = src.crs

    missing_data_mask = np.isclose(pop_map, nodata_value)
    pop_map = np.where(missing_data_mask, np.nan, pop_map)
    return pop_map, src


def load_boundaries_map():
    boundaries = gpd.read_file('ref_maps/world-administrative-boundaries/world-administrative-boundaries.shp')
    boundaries = boundaries.to_crs(epsg=4326)
    return boundaries


def calculate_impact(geo_df, pop_map, src, boundaries, date):
    geo_df = geo_df.to_crs(epsg=4326)
    df = pd.DataFrame({'regions': [], 'expected_impact': [], 'date': []})

    for storm in geo_df['cluster'].unique():
        storm_points = geo_df[geo_df['cluster'] == storm]
        pixel_values = []
        pixel_strike_prob = []
        for _, point in storm_points.iterrows():
            row, col = src.index(point.geometry.x, point.geometry.y)
            if not np.isnan(pop_map[row, col]):
                pixel_values.append(pop_map[row, col])
                pixel_strike_prob.append(point['strike_probability'])

        affected_pixels = np.unique(pixel_values)
        join_gdf = gpd.sjoin(storm_points, boundaries, how='inner', predicate='within')
        unique_admin_regions = join_gdf['name'].unique()
        expected_impact = sum([pop * strike_prob for pop, strike_prob in zip(affected_pixels, pixel_strike_prob)])
        new_row = pd.DataFrame({'regions': [unique_admin_regions], 'expected_impact': [expected_impact], 'date': [date]})
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def prepare_results(df_all, storm_type_options):
    df_all = df_all.dropna()
    df_all = df_all[df_all['expected_impact'] >= 100]
    return df_all
