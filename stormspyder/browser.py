import os
import shutil

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
        chrome_options.binary_location = _find_chrome_binary()
        service = ChromeService(ChromeDriverManager().install())

    return service, chrome_options


def get_webdriver(local_testing=False):
    service, chrome_options = build_browser_settings(local_testing=local_testing)
    return webdriver.Chrome(service=service, options=chrome_options), service, chrome_options


def _find_chrome_binary():
    # Allow explicit override first for CI or custom installs.
    env_path = os.environ.get("CHROME_BIN")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # Fall back to PATH lookups as a last resort.
    for binary_name in ["google-chrome-stable", "google-chrome", "chromium-browser", "chromium"]:
        discovered = shutil.which(binary_name)
        if discovered:
            return discovered

    raise RuntimeError("Chrome binary not found. Install Chrome/Chromium or set CHROME_BIN.")
