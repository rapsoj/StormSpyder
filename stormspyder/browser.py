import os

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:  # pragma: no cover - exercised in minimal environments
    webdriver = None
    ChromeService = None
    ChromeDriverManager = None
    WebDriverWait = None
    EC = None
    Options = None
    By = None
    TimeoutException = NoSuchElementException = Exception


def build_browser_settings(local_testing=False):
    if webdriver is None or ChromeService is None or Options is None or ChromeDriverManager is None:
        raise RuntimeError('Selenium and webdriver-manager are required to download forecast images')

    if local_testing:
        base_path = ChromeDriverManager().install()
        base_dir = os.path.dirname(base_path)
        service = ChromeService(executable_path=f"{base_dir}/chromedriver")
        chrome_options = webdriver.ChromeOptions()
    else:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.binary_location = "/usr/bin/google-chrome-stable"
        service = ChromeService(ChromeDriverManager().install())

    return service, chrome_options


def get_webdriver(local_testing=False):
    service, chrome_options = build_browser_settings(local_testing=local_testing)
    return webdriver.Chrome(service=service, options=chrome_options), service, chrome_options
