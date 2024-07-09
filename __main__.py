# Load system libraries
import os
import time
import requests
import shutil
from datetime import datetime

# Load data manipulation libraries
import numpy as np
import pandas as pd

# Load web-scraping libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Load image manipulation libraries
from PIL import Image
from scipy.ndimage import label
from scipy.spatial.distance import cdist
import rasterio

# Load geospatial libraries
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPoint

# Load machine learning libraries
from sklearn.cluster import DBSCAN

# Load email libraries
import yagmail

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.binary_location = "/usr/bin/google-chrome-stable"
chromedriver_path = "/usr/bin/chromedriver"


# Define function to download images from ECMWF website
def download_images(storm_type):

    # Path to the directory where you want to save the images
    save_dir = "temp/" + storm_type
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # URL of the website
    url = "https://charts.ecmwf.int/products/medium-tc-genesis"
    
    # XPaths for the relevant interactions
    dimensions_xpath = '//span[contains(text(), "Select dimensions")]'
    dropdown_xpath = '//div[@id="mui-component-select-layer_name"]'
    storm_type_xpath_template = '//li[@class="MuiButtonBase-root MuiListItem-root MuiMenuItem-root MuiMenuItem-gutters MuiListItem-gutters MuiListItem-button" and @data-value="{}"]'
    close_xpath = '//span[@class="MuiButton-label" and text()="Close"]'
    image_xpath_template = '//*[@id="root"]/div/div/div/div[3]/div/div[2]/div[1]/div/div/div/div[2]/img[{}]'
    next_button_xpath = '//*[@id="root"]/div/div/div/div[3]/div/div[2]/div[1]/div/div/div/div[3]/div[2]/div/button[4]'

    # Initialize the webdriver
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)
    
    # Wait for the page to load
    time.sleep(1)

    if storm_type != 'genesis_ts':
        try:

            # Wait for the dimensions button to be clickable and click it
            select_dimensions_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, dimensions_xpath)))
            select_dimensions_button.click()
        
            # Wait for the dropdown to be clickable and click it
            dropdown = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, dropdown_xpath)))
            dropdown.click()
        
            # Find and click on the storm type option
            storm_type_xpath = storm_type_xpath_template.format(storm_type)
            storm_type_element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, storm_type_xpath)))
            storm_type_element.click()
        
            # Wait for the close button to be clickable and click it
            close_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, close_xpath)))
            close_button.click()
    
            # Wait for the page to load after selecting options
            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, image_xpath_template.format(1))))
    
        except (TimeoutException, NoSuchElementException) as e:
            print(f"An error occurred: {e}")
            driver.quit()
            raise
    
    # Loop to download each image
    for i in range(1, 10):  # Assuming there are 9 images
        try:
            
            # Construct the XPath for the current image
            image_xpath = image_xpath_template.format(i)
    
            # Wait for the image element to be visible
            image_element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, image_xpath)))
            
            # Get the image source URL
            image_url = image_element.get_attribute("src")
            
            # Get the alt attribute of the image
            alt_text = image_element.get_attribute("alt")
            
            # Create a filename using the alt text
            image_filename = os.path.join(save_dir, f"{alt_text}.png")
            
            # Download the image
            image_path = os.path.join(save_dir, f"{alt_text}.png")
            with open(image_path, "wb") as f:
                f.write(requests.get(image_url).content)
            
            # Click the next button
            next_button = driver.find_element(By.XPATH, next_button_xpath)
            next_button.click()
            
            # Wait for the page to load
            time.sleep(1)

        except (TimeoutException, NoSuchElementException) as e:
            print(f"An error occurred while processing image {i}: {e}")
            break
        
    # Close the webdriver
    driver.quit()


# Define function to get image paths
def get_image_paths(storm_type):
    folder_path = 'temp/' + storm_type

    # Get list of all files and directories in the folder
    all_items = os.listdir(folder_path)
    
    # Filter out directories, only keeping files
    files = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
    
    return files


# Define function to convert pixel indices to geospatial coordinates
def pixel_to_geospatial(row, col, transform):
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


# Define function to check colour similarity
def is_similar_color(color1, color2, tolerance):
    return np.linalg.norm(np.array(color1) - np.array(color2)) < tolerance


# Define function to cluster points
def cluster_points(geo_df):
    if len(geo_df) > 0:
        
        # Convert spatial points into list
        points = geo_df.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
        
        # Set the minimum distance
        min_distance = 33392.607
        
        # Convert the GeoDataFrame points to a numpy array
        points = np.array(geo_df.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
        
        # Define the epsilon parameter as ten times the minimum distance
        epsilon = 5 * min_distance
        
        # Initialize DBSCAN clustering algorithm
        dbscan = DBSCAN(eps=epsilon, min_samples=1)
        
        # Fit DBSCAN to the points
        dbscan.fit(points)
        
        # Get labels assigned by DBSCAN
        labels = dbscan.labels_
        
        # Add cluster labels to the GeoDataFrame
        geo_df['cluster'] = labels
        
        # Count the number of points in each cluster
        cluster_counts = geo_df['cluster'].value_counts()
        
        # Get the cluster IDs with at least four points
        valid_clusters = cluster_counts[cluster_counts >= 4].index
        
        # Filter out points in clusters with less than four points
        geo_df = geo_df[geo_df['cluster'].isin(valid_clusters)]

        return geo_df


# Create function to detect strike probabilities from downloaded images
def detect_strike_probability(storm_type, date):

    # Loop through strike probabilities using associated colours and colour tolerance
    strike_probabilities = {0.05: (255,0,255,170), 0.15: (255,78,0,120), 0.25: (255,176,0,90), 0.35: (255,255,0,70), 0.45: (40,255,0,150),
                            0.55: (0,140,47,50), 0.65: (2,255,255,100), 0.75: (2,127,254,50), 0.85: (0,0,255,50), 0.95: (122,16,178,20)}
    
    # Load the original map image
    original_image = Image.open('temp/' + storm_type + '/' + date + '.png')
    
    # Convert the original image to numpy array
    original_array = np.array(original_image)
    
    # Load the georeferenced TIF
    georef_tif = 'ref_maps/georeferenced_map.tif'
    dataset = rasterio.open(georef_tif)
    
    # Get the transform from the georeferenced TIF
    transform = dataset.transform
    
    # Initialize an empty list to collect GeoDataFrames
    geo_df_list = []
    
    # Loop through strike probabilities
    for strike_prob in strike_probabilities.keys():
        
        # Define the strike probability target color and tolerance
        target_color = strike_probabilities[strike_prob][:3]
        color_tolerance = strike_probabilities[strike_prob][3]
        
        # Find the pixels that are similar to the target color
        matching_pixels = np.zeros((original_array.shape[0], original_array.shape[1]), dtype=bool)
        for row in range(original_array.shape[0]):
            for col in range(original_array.shape[1]):
                pixel_color = original_array[row, col, :3]
                if is_similar_color(pixel_color, target_color, color_tolerance):
                    matching_pixels[row, col] = True
        
        # Find the indices of the matching pixels
        pixel_indices = np.argwhere(matching_pixels)
        
        # Convert pixel indices to geospatial coordinates
        geospatial_points = [pixel_to_geospatial(row, col, transform) for row, col in pixel_indices]
        
        # Create a GeoDataFrame from the geospatial points
        geometry = [Point(x, y) for x, y in geospatial_points]
        geo_df_strike = gpd.GeoDataFrame(geometry=geometry, crs=dataset.crs)
        
        # Identify storm type and strike probability
        geo_df_strike['type'] = storm_type
        geo_df_strike['strike_probability'] = strike_prob
    
        # Append the current GeoDataFrame to the list
        geo_df_list.append(geo_df_strike)
    
    # Concatenate all GeoDataFrames in the list into a single GeoDataFrame
    geo_df = gpd.GeoDataFrame(pd.concat(geo_df_list, ignore_index=True), crs=dataset.crs)
    
    # Remove overlapping geometries by prioritising those with highest strike probability
    geo_df = geo_df.sort_values(by=['geometry', 'strike_probability'], ascending=[True, False])
    geo_df = geo_df.drop_duplicates(subset='geometry', keep='first')
    geo_df = geo_df.reset_index(drop=True)

    # Cluster points
    geo_df = cluster_points(geo_df)

    return geo_df


# Define function to load population map
def load_pop_map():
    # Open the raster file
    file_path = 'ref_maps/gpw_v4_population_count_rev11_2020_2pt5_min.tif'
    with rasterio.open(file_path) as src:
        # Read the raster data
        pop_map = src.read(1)  # Read the first band
        # Read the nodata value from metadata
        nodata_value = src.nodata
        # Get metadata
        metadata = src.meta
        # Get the coordinate reference system (CRS) of the raster
        raster_crs = src.crs
    
    # Mask missing data from population map
    missing_data_mask = np.isclose(pop_map, nodata_value)
    
    # Remove missing values from pop_map
    pop_map = np.where(missing_data_mask, np.nan, pop_map)

    return pop_map, src


# Define function to load boundaries map
def load_boundaries_map():
    
    # Read boundaries shapefile
    boundaries = gpd.read_file('ref_maps/world-administrative-boundaries/world-administrative-boundaries.shp')
    
    # Reproject data to EPSG:4326
    boundaries = boundaries.to_crs(epsg=4326)
    
    return boundaries


# Define function to calculate expected impact
def calculate_impact(geo_df, pop_map, src, boundaries, date):

    # Reproject data to EPSG:4326
    geo_df = geo_df.to_crs(epsg=4326)
    
    # Initialise dataframe
    df = pd.DataFrame({'regions': [], 'expected_impact': [], 'date': []})
    
    # Loop through storm clusters
    for storm in geo_df['cluster'].unique():
    
        # Extract pixel values corresponding to points
        storm_points = geo_df[geo_df['cluster'] == storm]
        pixel_values = []
        pixel_strike_prob = []
        for index, point in storm_points.iterrows():
            row, col = src.index(point.geometry.x, point.geometry.y)
            if not np.isnan(pop_map[row, col]):
                pixel_values.append(pop_map[row, col])
                pixel_strike_prob.append(point['strike_probability'])
        
        # Get unique pixel values affected by storm
        affected_pixels = np.unique(pixel_values)
    
        # Find which administrative regions are impacted by the storm
        join_gdf = gpd.sjoin(storm_points, boundaries, how="inner", predicate="within")
    
        # Extract the unique administrative regions
        unique_admin_regions = join_gdf['name'].unique()
    
        # Calculate expected impact
        expected_impact = sum([pop * strike_prob for pop, strike_prob in zip(affected_pixels, pixel_strike_prob)])
    
        # Create a new DataFrame for the current storm
        new_row = pd.DataFrame({'regions': [unique_admin_regions], 'expected_impact': [expected_impact], 'date': [date]})
    
        # Concatenate the new row to the existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df


# Define function to get dictionary key from value
def get_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key


# Define function to format regions for subject line
def format_regions(arr):
    length = len(arr)
    if length == 0:
        return "unclaimed territories"
    elif length == 1:
        return arr[0]
    elif length == 2:
        return f"{arr[0]} and {arr[1]}"
    elif length == 3:
        return f"{arr[0]}, {arr[1]}, and {arr[2]}"
    elif length == 4:
        return f"{arr[0]}, {arr[1]}, {arr[2]}, and one other country"
    else:
        return f"{arr[0]}, {arr[1]}, {arr[2]}, and {length - 3} other countries"


# Define function to format impact estimates for subject line
def format_impact(num):
    if num > 1000000:
        return str(round(num / 1000000, 1)) + "M"
    elif num > 1000:
        return str(round(num / 1000, 1)) + "K"
    else:
        return int(round(num, 0))


# Define function to format alert text
def format_alert(df_all, n):

    # Extract information for alert
    n_impact = df_all.sort_values('expected_impact', ascending=False).iloc[n]
    storm_type = n_impact['storm_type']
    regions = format_regions(n_impact['regions'])
    impact = format_impact(n_impact['expected_impact'])
    date = n_impact['date']

    # Combine information into alert
    alert = f"{storm_type.capitalize()} alert: {impact} estimated impacted in {regions} on {date}"

    return alert


# Define function to send email alert
def send_email_alert(df_all, storm_type_options):
    # Create dataframe for storm events that intersect with populated areas
    today_date = datetime.today().strftime('%Y-%m-%d')
    data = df_all.sort_values('expected_impact', ascending=False)
    data = data[data['expected_impact'] > 0]
    data['regions'] = data['regions'].apply(lambda x: ', '.join(x))
    data['reported'] = today_date
    data.to_csv(f"{today_date}-storm-alert.csv", index=False)

    # Extract information for subject line
    top_impact = df_all.sort_values('expected_impact', ascending=False).iloc[0]
    storm_type = top_impact['storm_type']
    regions = format_regions(top_impact['regions'])
    impact = format_impact(top_impact['expected_impact'])
    date = top_impact['date']

    # Set information to send email
    FROM = {'stormspyder.alerts@gmail.com': 'StormSpyder'}
    TO = ["jessicakristenr@gmail.com", "elisabeth.stephens@reading.ac.uk"] # Change this as needed
    #TO = ["jessicakristenr@gmail.com", "jess.rapson@rogers.com"] # For testing
    APP_PASSWORD = "blsz ardo uicy qfnx"
    SUBJECT = f"{storm_type.capitalize()} alert: {impact} estimated impacted in {regions} on {date}"
    IMAGE_PATH = 'temp/' + get_key_by_value(storm_type_options, storm_type) + '/' + date + '.png'
    CSV_PATH = f"{today_date}-storm-alert.csv"

    # Create the numbered list of alerts
    alert_list = "<ol>"
    for n in range(len(df_all)):
        alert = format_alert(df_all, n)
        alert_list += f"<li>{alert}</li>"
    alert_list += "</ol>"

    # Prepare the message
    message_html = f"""
    <html>
      <body>
        <p>Dear recipient,</p>
        <p>The following tropical storm events have been forecast by the ECMWF:</p>
        {alert_list}
        <p>{storm_type.capitalize()} forecast for {date}:</p>
        <img src="cid:image.png">
        <p>To learn more about how this tool works, <a href="https://github.com/rapsoj/StormSpyder/tree/main">click here</a>.</p>
        <p>To unsubscribe from future alerts, send a message to <a href="mailto:stormspyder.alerts@gmail.com?subject=UNSUBSCRIBE">stormspyder.alerts@gmail.com</a> with the subject line "UNSUBSCRIBE".</p>
        <p>Best regards,<br>StormSpyder</p>
      </body>
    </html>
    """

    # Send the email with attachments
    yag = yagmail.SMTP(FROM, APP_PASSWORD)
    yag.send(
        bcc=TO,
        subject=SUBJECT,
        contents=[message_html, yagmail.inline(IMAGE_PATH)],
        attachments=CSV_PATH
    )
    print("---Email sent successfully---")

    # Store daily results in database of historic events
    record = pd.read_csv('record.csv')
    if not record['reported'].isin([today_date]).any():
        record = pd.concat([record, data], ignore_index=True)
        record.to_csv('record.csv', index=False)

    # Delete today's file
    os.remove(CSV_PATH)


# Define main function
def main():

    print(datetime.now().time().strftime("%H:%M:%S"))

    # Load reference maps
    pop_map, src = load_pop_map()
    boundaries = load_boundaries_map()

    print("---Loaded reference maps---")

    # Initialise final dataframe
    df_all = pd.DataFrame({'regions': [], 'expected_impact': [], 'date': [], 'storm_type': []})

    # Delete temp folder if it exists
    if os.path.exists('temp') and os.path.isdir('temp'):
        # Delete the folder and all its contents
        shutil.rmtree('temp')

    # Loop through storm types
    storm_type_options = {'genesis_ts': 'tropical storm', 'genesis_hr': 'hurricane'}
    for storm_type in storm_type_options.keys():

        # Print progress
        print(f"Downloading {storm_type_options[storm_type]} images...")

        # Download images from ECMWF website
        download_images(storm_type)
        
        # Print progress
        print(f"Processing {storm_type_options[storm_type]} images...")
        
        # Loop through downloaded images
        for file in get_image_paths(storm_type):

            # Get date from file name
            date = file[:-4]
            
            # Print progress
            print(f"Detecting {storm_type_options[storm_type]} strike probabilities for {date}")

            # Detect strike probabilities for selected date
            geo_df = detect_strike_probability(storm_type, date)

            # Compare storm clusters with population map
            if geo_df is not None:
                df = calculate_impact(geo_df, pop_map, src, boundaries, date)
                df['storm_type'] = storm_type_options[storm_type]
        
                # Combine results into single dataframe
                df_all = pd.concat([df_all, df], ignore_index=True)

        print(datetime.now().time().strftime("%H:%M:%S"))
        
        # Print break
        print('---------------------------------------------------------------------------------')

    print(datetime.now().time().strftime("%H:%M:%S"))

    # Remove empty rows
    df_all = df_all.dropna()

    # Remove rows with less than 100 people estimated impacted
    df_all = df_all[df_all['expected_impact'] >= 100]

    # Send emails
    send_email_alert(df_all, storm_type_options)


# Execute tool
if __name__ == '__main__':
    main()