import os
import shutil
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

try:
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:  # pragma: no cover - exercised in minimal environments
    TimeoutException = NoSuchElementException = Exception
    EC = None
    By = None
    WebDriverWait = None

from .alerts import send_email_alert
from .processing import (
    calculate_impact,
    detect_strike_probability,
    get_image_paths,
    load_boundaries_map,
    load_pop_map,
    prepare_results,
)


OPENCHARTS_BASE = 'https://charts.ecmwf.int/opencharts-api/v1'
OPENCHARTS_AXIS_ENDPOINT = (
    f'{OPENCHARTS_BASE}/packages/opencharts/products/medium-tc-genesis/axis/'
)
OPENCHARTS_VALID_TIME_ENDPOINT = (
    f'{OPENCHARTS_BASE}/packages/opencharts/products/medium-tc-genesis/axis/valid_time/'
)


def _write_image_as_png(image_bytes, output_path):
    # ECMWF currently serves WebP frames; re-encoding to PNG keeps downstream file handling stable.
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert('RGB').save(output_path, format='PNG')


def _download_images_via_api(storm_type, save_dir):
    axis_response = requests.get(OPENCHARTS_AXIS_ENDPOINT, timeout=30)
    axis_response.raise_for_status()
    axis_payload = axis_response.json()
    axes = axis_payload.get('axis', [])
    base_time_axis = next((axis for axis in axes if axis.get('name') == 'base_time'), None)
    if base_time_axis is None or not base_time_axis.get('values'):
        raise RuntimeError('Unable to determine latest base_time from ECMWF axis metadata')

    latest_base_time = base_time_axis['values'][0]['value']

    response = requests.get(
        OPENCHARTS_VALID_TIME_ENDPOINT,
        params={
            'base_time': latest_base_time,
            'layer_name': storm_type,
            'projection': 'opencharts_global',
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get('results', {})

    if not results:
        raise RuntimeError(f'No chart results found for {storm_type}')

    downloaded_count = 0
    for valid_time, chart in sorted(results.items()):
        image_url = chart.get('url')
        if not image_url:
            continue

        image_response = requests.get(image_url, timeout=45)
        image_response.raise_for_status()

        image_path = os.path.join(save_dir, f'{valid_time}.png')
        _write_image_as_png(image_response.content, image_path)
        downloaded_count += 1

    if downloaded_count == 0:
        raise RuntimeError(f'No downloadable image URLs returned for {storm_type}')

    return downloaded_count


def download_images(storm_type, local_testing=False):
    save_dir = os.path.join('temp', storm_type)
    os.makedirs(save_dir, exist_ok=True)

    # Keep existing signature for compatibility; local_testing is not required for API mode.
    _ = local_testing
    try:
        downloaded = _download_images_via_api(storm_type, save_dir)
        print(f'Downloaded {downloaded} {storm_type} forecast images.')
    except Exception as e:
        raise RuntimeError(f'Failed downloading {storm_type} images from ECMWF API') from e


def run_pipeline(local_testing=False):
    print(datetime.now().time().strftime('%H:%M:%S'))
    pop_map, src = load_pop_map()
    boundaries = load_boundaries_map()
    print('---Loaded reference maps---')

    df_all = pd.DataFrame({'regions': [], 'expected_impact': [], 'date': [], 'storm_type': []})

    if os.path.exists('temp') and os.path.isdir('temp'):
        shutil.rmtree('temp')

    storm_type_options = {'genesis_ts': 'tropical storm', 'genesis_hr': 'hurricane'}
    for storm_type in storm_type_options.keys():
        print(f"Downloading {storm_type_options[storm_type]} images...")
        download_images(storm_type, local_testing=local_testing)

        print(f"Processing {storm_type_options[storm_type]} images...")
        for file in get_image_paths(storm_type):
            date = file[:-4]
            print(f"Detecting {storm_type_options[storm_type]} strike probabilities for {date}")
            geo_df = detect_strike_probability(storm_type, date)
            if geo_df is not None:
                df = calculate_impact(geo_df, pop_map, src, boundaries, date)
                df['storm_type'] = storm_type_options[storm_type]
                df_all = pd.concat([df_all, df], ignore_index=True)

        print(datetime.now().time().strftime('%H:%M:%S'))
        print('---------------------------------------------------------------------------------')

    print(datetime.now().time().strftime('%H:%M:%S'))
    df_all = prepare_results(df_all, storm_type_options)
    return df_all, storm_type_options


def run_pipeline_with_email(local_testing=False):
    df_all, storm_type_options = run_pipeline(local_testing=local_testing)
    send_email_alert(df_all, storm_type_options)
