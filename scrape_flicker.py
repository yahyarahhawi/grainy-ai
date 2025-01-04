import os
import time
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    StaleElementReferenceException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def create_chrome_driver(chromedriver_path):
    """
    Create and return a headless Chrome driver with desired options.
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = webdriver.chrome.service.Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def scroll_to_position(driver, position):
    """
    Scroll the browser window to a given vertical position.
    """
    driver.execute_script(f"window.scrollTo(0, {position});")


def save_image(url, idx, save_dir):
    """
    Download and save an image from a URL to the specified directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            image_path = os.path.join(save_dir, f"{idx}.jpg")
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {idx}: {url}")
        else:
            print(f"Failed to download (status {response.status_code}): {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def click_flickr_image(driver, element):
    """
    Use WebDriverWait + JS-based click to avoid interception.
    Scroll the element into view, wait until clickable, then click via JS.
    """
    wait = WebDriverWait(driver, 10)

    # Scroll to the element
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    time.sleep(1)

    # Wait until clickable
    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.overlay")))

    # Attempt JS click
    driver.execute_script("arguments[0].click();", element)
    time.sleep(2)  # Short wait for navigation/transition


def scrape_flickr_search(start_url,
                        max_images=100,
                        chromedriver_path="path/to/chromedriver",
                        save_dir="flickr_full_images",
                        sleep_between=2,
                        retries_on_crash=3):
    """
    Scrape Flickr search results:
      - Keep track of downloaded images (image_count).
      - If browser/tab crashes, re-initialize driver and scroll to
        approximately where we left off, up to 'retries_on_crash' times.
      - Use JavaScript-based click to handle click interception errors.
    """

    image_count = 0
    retry_count = 0
    current_scroll_position = 0

    while image_count < max_images and retry_count < retries_on_crash:
        driver = None
        try:
            # Initialize (or re-initialize) the driver
            driver = create_chrome_driver(chromedriver_path)
            driver.get(start_url)
            time.sleep(sleep_between)

            # Scroll down to approximate position
            scroll_to_position(driver, current_scroll_position)
            time.sleep(sleep_between)

            # Main scraping loop
            while image_count < max_images:
                # Fetch all currently loaded image links
                image_links = driver.find_elements(By.CSS_SELECTOR, "a.overlay")

                # If we've processed everything loaded so far, try loading more
                if image_count >= len(image_links):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(sleep_between + 2)

                    # Try clicking 'Load more results' button
                    try:
                        load_more_button = driver.find_element(By.XPATH,
                            "//button[contains(text(), 'Load more results')]")
                        load_more_button.click()
                        print("Clicked 'Load more results'. Waiting for more images to load...")
                        time.sleep(sleep_between + 3)
                    except NoSuchElementException:
                        print("No 'Load more results' button found or no more images to load.")
                        break
                    continue

                # Process the next image
                try:
                    # Attempt to click with JS-based logic
                    click_flickr_image(driver, image_links[image_count])

                    # Extract full-size image URL
                    full_image = driver.find_element(By.CSS_SELECTOR, "img.main-photo")
                    src = full_image.get_attribute("src")

                    if src:
                        image_count += 1
                        save_image(src, image_count, save_dir)
                        # Update approximate scroll position after each download
                        current_scroll_position += 300

                    # Click the "Back to search" or go back
                    try:
                        back_to_search = driver.find_element(By.CSS_SELECTOR, "a.entry-type")
                        driver.execute_script("arguments[0].click();", back_to_search)
                        time.sleep(sleep_between)
                    except NoSuchElementException:
                        print("Could not find 'Back to search' link. Going back via browser.")
                        driver.back()
                        time.sleep(sleep_between)

                    # Scroll back to approximate position on the search results page
                    scroll_to_position(driver, current_scroll_position)
                    time.sleep(sleep_between // 2)

                except (NoSuchElementException, StaleElementReferenceException) as e:
                    # If an error arises for a specific image, skip it
                    print(f"Error processing image {image_count + 1}: {e}")
                    image_count += 1
                    time.sleep(sleep_between // 2)

            print("Scraping complete (or no more images to load).")
            driver.quit()
            break  # Completed scraping without a crash

        except WebDriverException as wde:
            print(f"WebDriverException encountered: {wde}")
            print("Reinitializing the driver and resuming...")
            retry_count += 1

            # Close the crashed driver if it's still alive
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            time.sleep(5)

    if image_count >= max_images:
        print(f"Successfully downloaded {image_count} images.")
    else:
        print(f"Stopped after downloading {image_count} images with {retry_count} crashes.")


if __name__ == "__main__":
    scrape_flickr_search(
        start_url="https://www.flickr.com/search/?text=cinestill+800t&view_all=1",
        max_images=1000,
        chromedriver_path="/Users/yahyarahhawi/Downloads/chromedriver-mac-arm64/chromedriver",
        save_dir="new_dataset"
    )