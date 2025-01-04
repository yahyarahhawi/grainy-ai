from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException
)
import time
import requests
import os

def scrape_flickr_paginated(
    base_url,
    max_images=100,
    chromedriver_path="path/to/chromedriver",
    save_dir="flickr_full_images",
    start_page=1,
):
    """
    Scrape images from paginated Flickr group pages.
    e.g., https://www.flickr.com/groups/cinestillfilm/pool/page1, /page2, ...
    
    :param base_url: The base URL without the trailing slash and page number.
    :param max_images: Maximum number of images to download.
    :param chromedriver_path: Path to the ChromeDriver executable.
    :param save_dir: Directory where downloaded images will be saved.
    :param start_page: The first page to start scraping from.
    """

    # Setup Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize WebDriver
    service = webdriver.chrome.service.Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    image_count = 0
    page_num = start_page

    try:
        while image_count < max_images:
            # Construct the paginated URL
            page_url = f"{base_url}/page{page_num}"
            print(f"Navigating to: {page_url}")
            driver.get(page_url)
            time.sleep(3)

            # Get all <a> elements that link to the image detail page
            image_links = driver.find_elements(By.CSS_SELECTOR, "a.overlay")

            # If no images found, we assume we've gone beyond the last page
            if not image_links:
                print(f"No images found on page {page_num}. Stopping.")
                break

            # Collect all the link URLs first to avoid stale references
            link_urls = [link.get_attribute("href") for link in image_links if link.get_attribute("href")]

            # Visit each link's detail page to scrape the full-size image
            for link_url in link_urls:
                if image_count >= max_images:
                    break

                # Navigate directly to the image detail page
                try:
                    driver.get(link_url)
                    time.sleep(2)

                    # Attempt to find the large "main-photo" element
                    try:
                        full_image = driver.find_element(By.CSS_SELECTOR, "img.main-photo")
                        src = full_image.get_attribute("src")
                    except NoSuchElementException:
                        # If there's no "main-photo" on this page, skip to the next link
                        print(f"No main-photo found for image {image_count + 1} at {link_url}.")
                        continue

                    # Save image if src is valid
                    if src:
                        image_count += 1
                        save_image(src, image_count, save_dir)

                except TimeoutException:
                    print(f"Timeout loading {link_url}. Skipping.")
                    continue
                except Exception as e:
                    print(f"Unexpected error loading {link_url}: {e}")
                    continue

            # Move on to the next page
            page_num += 1

    finally:
        driver.quit()


def save_image(url, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    try:
        response = requests.get(url, stream=True)
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


if __name__ == "__main__":
    scrape_flickr_paginated(
        base_url="https://www.flickr.com/groups/cinestillfilm/pool",
        max_images=1000,
        chromedriver_path="/Users/yahyarahhawi/Downloads/chromedriver-mac-arm64/chromedriver",
        save_dir="new_dataset2",
        start_page=1
    )