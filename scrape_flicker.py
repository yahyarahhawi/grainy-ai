import time
import os
import requests
import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    StaleElementReferenceException,
    NoSuchElementException
)

def create_chrome_driver(chromedriver_path=None, headless=True):
    """
    Create and return a Chrome WebDriver instance.
    Set headless=True to run without opening a browser window.
    """
    chrome_options = webdriver.ChromeOptions()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    if chromedriver_path:
        service = webdriver.chrome.service.Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
    else:
        # This requires chromedriver on your PATH
        driver = webdriver.Chrome(options=chrome_options)

    return driver


def lazy_scroll(driver, pause_time=2, max_scroll_attempts=3):
    """
    Scroll the page until no new content is loaded,
    or until we've done 'max_scroll_attempts' consecutive scrolls without progress.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0

    while True:
        # Scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            scroll_attempts += 1
        else:
            scroll_attempts = 0

        if scroll_attempts >= max_scroll_attempts:
            print("Reached the end of scrolling or no more new content.")
            break

        last_height = new_height


def save_image(url, idx, save_dir="reddit_cinestill_images"):
    """
    Download the image from 'url' and save to 'save_dir' with index 'idx'.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{idx}.jpg")

    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {idx}: {url}")
        else:
            print(f"Failed to download (status {response.status_code}): {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def scrape_reddit_images(
    start_url="https://www.reddit.com/r/analog/search/?q=cinestill&cId=39049d8c-d493-4d36-a334-1cfec7887ab7&iId=9a235081-9359-48f2-8a00-e360e64a7368",
    max_images=100,
    chromedriver_path="/path/to/chromedriver",
    headless=True,
    save_dir="reddit_cinestill_images"
):
    driver = create_chrome_driver(chromedriver_path=chromedriver_path, headless=headless)

    try:
        driver.get(start_url)
        time.sleep(3)  # Let the page load a bit

        # Scroll to load all possible posts
        lazy_scroll(driver, pause_time=2, max_scroll_attempts=3)

        # Collect all 'thumbnail' images: these typically have a 'thumbs.redditmedia.com' src
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img[src*='thumbs.redditmedia.com']")
        print(f"Found {len(thumbnails)} potential thumbnails.")

        image_count = 0
        for i, thumb in enumerate(thumbnails):
            if image_count >= max_images:
                break

            try:
                # Click the thumbnail to open lightbox
                thumb.click()
                time.sleep(2)

                # Find the large image in the lightbox:
                # It's usually <img class="media-lightbox-img" src="...">
                # We'll attempt it with a small retry for staleness or no element
                attempts = 0
                while attempts < 3:
                    try:
                        full_img = driver.find_element(By.CSS_SELECTOR, "img.media-lightbox-img")
                        src = full_img.get_attribute("src")
                        break
                    except StaleElementReferenceException:
                        attempts += 1
                        time.sleep(1)
                        print(f"Retry find_element for full_img. Attempt {attempts}")

                if src:
                    image_count += 1
                    save_image(src, image_count, save_dir=save_dir)
                else:
                    print("No image src found in lightbox; skipping.")

                # Close the lightbox (send ESC) or find a close button
                driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                time.sleep(1)

            except NoSuchElementException:
                print("No lightbox or no large image found; skipping this thumbnail.")
            except Exception as e:
                print(f"Error processing thumbnail {i+1}: {e}")

    finally:
        driver.quit()
        print(f"Done. Downloaded {image_count} images to '{save_dir}'.")


if __name__ == "__main__":
    scrape_reddit_images(
        start_url=(
            "https://www.reddit.com/r/analog/search/?q=cinestill"
            "&cId=39049d8c-d493-4d36-a334-1cfec7887ab7"
            "&iId=9a235081-9359-48f2-8a00-e360e64a7368"
        ),
        max_images=200,  # or however many you want
        chromedriver_path="/Users/yahyarahhawi/Downloads/chromedriver-mac-arm64/chromedriver",
        headless=True,
        save_dir="cinestill_reddit_images"
    )