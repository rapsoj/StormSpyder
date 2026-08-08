import os
import shutil
from datetime import datetime

import pandas as pd
import requests

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
from .browser import get_webdriver
from .processing import (
    calculate_impact,
    detect_strike_probability,
    get_image_paths,
    load_boundaries_map,
    load_pop_map,
    prepare_results,
)


def download_images(storm_type, local_testing=False):
    save_dir = os.path.join('temp', storm_type)
    os.makedirs(save_dir, exist_ok=True)

    url = f"https://charts.ecmwf.int/products/medium-tc-genesis?layer_name={storm_type}"
    image_xpath_template = '//*[@id="root"]/div/div/div/div[3]/div/div[2]/div[1]/div/div/div/div[2]/img[{}]'
    next_button_xpath = '//*[@id="root"]/div/div/div/div[3]/div/div[2]/div[1]/div/div/div/div[3]/div[2]/div/button[4]'

    driver, _, _ = get_webdriver(local_testing=local_testing)
    driver.get(url)

    for i in range(1, 10):
        try:
            image_xpath = image_xpath_template.format(i)
            image_element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, image_xpath)))
            image_url = image_element.get_attribute('src')
            alt_text = image_element.get_attribute('alt')
            image_path = os.path.join(save_dir, f'{alt_text}.png')
            with open(image_path, 'wb') as f:
                f.write(requests.get(image_url).content)

            next_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
            next_button.click()
        except (TimeoutException, NoSuchElementException) as e:
            print(f'An error occurred while processing image {i}: {e}')
            break

    driver.quit()


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
