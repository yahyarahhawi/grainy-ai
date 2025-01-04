from bs4 import BeautifulSoup
import requests
import os
import sys

def scrape_lomography(start_page=1, end_page=10):
    # Base URL for the website
    base_url = "https://www.lomography.com"

    # Headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # Create a folder to save the images
    os.makedirs('lomography_images', exist_ok=True)

    # Counter for naming the images sequentially
    image_index = 3833

    for page in range(start_page, end_page + 1):
        # Construct the URL for the current page
        html_page = f"{base_url}/films/871910984-kodak-portra-400/photos?page={page}"
        print(f"Scraping page {page}: {html_page}")

        # Make a GET request to the current page
        response = requests.get(html_page, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the page content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links to individual photo pages
            photo_links = [base_url + link.get('href') for link in soup.find_all('a', href=True) if "/photos/" in link.get('href')]

            # Remove duplicates
            photo_links = list(set(photo_links))

            for photo_url in photo_links:
                print(f"Visiting: {photo_url}")
                try:
                    # Visit each photo page
                    photo_response = requests.get(photo_url, headers=headers)

                    # Check if the request was successful
                    if photo_response.status_code == 200:
                        # Parse the photo page
                        photo_soup = BeautifulSoup(photo_response.text, 'html.parser')

                        # Find the full-size image URL
                        full_img_tag = photo_soup.find('img', {'class': ''})  # Update the class if needed
                        if full_img_tag and full_img_tag.get('src'):
                            full_img_url = full_img_tag.get('src')

                            # Skip images containing 'subscribe' in the URL
                            if 'subscribe' in full_img_url:
                                print(f"Skipping subscription image: {full_img_url}")
                                continue

                            # Download the full-size image
                            print(f"Downloading: {full_img_url}")
                            img_response = requests.get(full_img_url, headers=headers, stream=True)

                            if img_response.status_code == 200:
                                # Save the image in the folder with a sequential name
                                with open(f'lomography_images/{image_index}.jpg', "wb") as file:
                                    for chunk in img_response.iter_content(1024):
                                        file.write(chunk)
                                image_index += 1  # Increment the image index
                            else:
                                print(f"Failed to download {full_img_url}: {img_response.status_code}")
                    else:
                        print(f"Failed to fetch the photo page: {photo_url}")
                except Exception as e:
                    print(f"Error processing {photo_url}: {e}")
        else:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")

if __name__ == "__main__":
    # Allow command-line arguments for start and end pages
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Call the scraping function with the specified or default arguments
    scrape_lomography(start, end)