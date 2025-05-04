"""
Web scraper for images from online sources, optimized for educational purposes
"""
import os
import time
import random
import requests
import hashlib
from PIL import Image
from io import BytesIO
import urllib.parse
import re
import sys
import traceback

from utils import config

def get_image_hash(image):
    """Generate a hash for image content to detect duplicates"""
    image = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = [1 if pixel > avg else 0 for pixel in pixels]
    hex_hash = ''.join([hex(int(''.join(map(str, bits[i:i+4])), 2))[2:] 
                     for i in range(0, len(bits), 4)])
    return hex_hash

def try_fix_url(url):
    """Attempt to fix common URL encoding issues in image URLs"""
    # Fix double-escaped URLs
    url = url.replace('\\u002F', '/').replace('\\/', '/')
    
    # Remove any escaped quotes
    url = url.replace('\\"', '"').replace('\"', '"')
    
    # Fix protocol if needed
    if url.startswith('//'):
        url = 'https:' + url
        
    return url

def scrape_images(search_term, output_dir, num_images=None):
    """
    Scrapes images from Yandex based on search term
    
    Args:
        search_term (str): The search term to use
        output_dir (str): Directory to save downloaded images
        num_images (int): Maximum number of images to download
    
    Returns:
        int: Number of successfully downloaded images
    """
    if num_images is None:
        num_images = config.DEFAULT_NUM_IMAGES
        
    print(f"Scraping Yandex for '{search_term}'...")
    
    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Format search term for URL
    encoded_query = urllib.parse.quote(search_term)
    
    # Build the Yandex search URL (adding adult content parameter if needed)
    search_url = f"https://yandex.com/images/search?text={encoded_query}"
    
    # For NSFW searches, add the adult content parameter
    if any(term in search_term.lower() for term in ['nsfw', 'adult', 'xxx']):
        search_url += "&noreask=1&adult=1"
    
    # Set up headers to avoid blocking
    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://yandex.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        # Get the search page
        print(f"Requesting search URL: {search_url}")
        response = requests.get(search_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
        html_content = response.text
        
        # Store this for debugging
        debug_file = os.path.join(output_dir, "yandex_response.html")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved response HTML to {debug_file} for debugging")
        
        # Extract image URLs using several methods
        image_urls = []
        
        # Method 1: Look for JSON data in the page that contains image URLs
        json_matches = re.findall(r'"url":"(https?:\\?/\\?/[^"]+\.(jpg|jpeg|png))"', html_content)
        for match in json_matches:
            url = try_fix_url(match[0])
            if url not in image_urls and url.startswith('http'):
                image_urls.append(url)
        
        # Method 2: Look for regular image tags
        img_matches = re.findall(r'<img[^>]+src=["\'](https?://[^"\']+\.(jpg|jpeg|png))["\']', html_content)
        for match in img_matches:
            if match[0] not in image_urls:
                image_urls.append(match[0])
                
        # Method 3: Look for background images in style attributes
        bg_matches = re.findall(r'background-image:\s*url\(["\']?(https?://[^"\']+\.(jpg|jpeg|png))["\']?\)', html_content)
        for match in bg_matches:
            if match[0] not in image_urls:
                image_urls.append(match[0])
        
        # Method 4: Look for the Yandex-specific data structure
        serp_items = re.findall(r'data-bem=[\'"]({\\?"serp-item\\?":.+?})[\'"]', html_content)
        for item in serp_items:
            # Look for image URLs in the data-bem attribute
            img_match = re.search(r'(?:imgUrl|img_url|originUrl|origin_url|url)[\"\']\s*:\s*[\"\'](https?:\\?/\\?/[^\"\']+\.(jpg|jpeg|png))[\"\']\s*[,}]', item)
            if img_match:
                url = try_fix_url(img_match.group(1))
                if url not in image_urls and url.startswith('http'):
                    image_urls.append(url)
        
        print(f"Found {len(image_urls)} image URLs")
        
        # If we still don't have any URLs, try a fallback method
        if not image_urls:
            print("Using direct API approach for Yandex")
            # Direct Yandex API approach (less likely to work but worth trying)
            api_url = f"https://yandex.com/images/touch/search?text={encoded_query}"
            try:
                api_resp = requests.get(api_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
                # Look for image URLs in the JSON response
                api_urls = re.findall(r'"url":"(https?:\\?/\\?/[^"]+\.(jpg|jpeg|png))"', api_resp.text)
                for match in api_urls:
                    url = try_fix_url(match[0])
                    if url not in image_urls and url.startswith('http'):
                        image_urls.append(url)
                print(f"API approach found {len(image_urls)} URLs")
            except Exception as e:
                print(f"API approach failed: {e}")
        
        # Keep track of image hashes to avoid duplicates
        image_hashes = set()
        downloaded_count = 0
        
        # Save a debug list of all found URLs
        with open(os.path.join(output_dir, "image_urls.txt"), 'w') as f:
            for url in image_urls:
                f.write(url + '\n')
        
        # Download images
        for i, img_url in enumerate(image_urls):
            if downloaded_count >= num_images:
                break
                
            try:
                # Add a small delay to avoid being blocked
                time.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))
                
                # Try to download the image
                print(f"Downloading: {img_url}")
                img_response = requests.get(img_url, headers=headers, timeout=10)
                
                if img_response.status_code == 200:
                    try:
                        # Check if it's a valid image
                        img = Image.open(BytesIO(img_response.content))
                        
                        # Handle the transparency warning by explicitly converting if needed
                        if img.mode == 'P' and 'transparency' in img.info:
                            img = img.convert('RGBA')
                        
                        # Check for duplicates using image hashing
                        img_hash = get_image_hash(img)
                        if img_hash in image_hashes:
                            print(f"Skipping duplicate image")
                            continue
                        
                        image_hashes.add(img_hash)
                        
                        # Save the image
                        file_extension = img.format.lower() if img.format else "jpg"
                        img_path = os.path.join(output_dir, f"{search_term.replace(' ', '_')}_{downloaded_count}.{file_extension}")
                        img.save(img_path)
                        
                        print(f"Downloaded: {img_path}")
                        downloaded_count += 1
                        
                    except Exception as e:
                        print(f"Invalid image format: {e}")
                else:
                    print(f"Failed to download: Status code {img_response.status_code}")
                
            except Exception as e:
                print(f"Error downloading image: {e}")
        
        # If we didn't get enough images from the initial page, try another approach
        if downloaded_count < num_images:
            # Try to get more image URLs from other pages
            for page in range(2, 5):  # Try a few more pages
                if downloaded_count >= num_images:
                    break
                
                next_page_url = f"{search_url}&p={page}"
                try:
                    print(f"Requesting page {page}: {next_page_url}")
                    response = requests.get(next_page_url, headers=headers, timeout=config.REQUEST_TIMEOUT)
                    html_content = response.text
                    
                    # Extract more image URLs
                    new_urls = []
                    
                    json_matches = re.findall(r'"url":"(https?:\\?/\\?/[^"]+\.(jpg|jpeg|png))"', html_content)
                    for match in json_matches:
                        url = try_fix_url(match[0])
                        if url not in image_urls and url not in new_urls and url.startswith('http'):
                            new_urls.append(url)
                    
                    serp_items = re.findall(r'data-bem=[\'"]({\\?"serp-item\\?":.+?})[\'"]', html_content)
                    for item in serp_items:
                        img_match = re.search(r'(?:imgUrl|img_url|originUrl|origin_url|url)[\"\']\s*:\s*[\"\'](https?:\\?/\\?/[^\"\']+\.(jpg|jpeg|png))[\"\']\s*[,}]', item)
                        if img_match:
                            url = try_fix_url(img_match.group(1))
                            if url not in image_urls and url not in new_urls and url.startswith('http'):
                                new_urls.append(url)
                    
                    print(f"Found {len(new_urls)} new URLs on page {page}")
                    image_urls.extend(new_urls)
                    
                    # Process these new URLs
                    for img_url in new_urls:
                        if downloaded_count >= num_images:
                            break
                            
                        try:
                            time.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))
                            print(f"Downloading: {img_url}")
                            img_response = requests.get(img_url, headers=headers, timeout=10)
                            
                            if img_response.status_code == 200:
                                img = Image.open(BytesIO(img_response.content))
                                
                                # Handle transparency
                                if img.mode == 'P' and 'transparency' in img.info:
                                    img = img.convert('RGBA')
                                    
                                img_hash = get_image_hash(img)
                                
                                if img_hash in image_hashes:
                                    print("Skipping duplicate image")
                                    continue
                                    
                                image_hashes.add(img_hash)
                                file_extension = img.format.lower() if img.format else "jpg"
                                img_path = os.path.join(output_dir, f"{search_term.replace(' ', '_')}_{downloaded_count}.{file_extension}")
                                img.save(img_path)
                                
                                print(f"Downloaded: {img_path}")
                                downloaded_count += 1
                                
                        except Exception as e:
                            print(f"Error downloading image: {e}")
                
                except Exception as e:
                    print(f"Error fetching page {page}: {e}")
        
        # Use direct Unsplash API as a fallback if we still don't have images
        if downloaded_count < num_images:
            try:
                print("Trying alternative source: Unsplash")
                for i in range(min(num_images - downloaded_count, 5)):
                    unsplash_url = f"https://source.unsplash.com/random/800x600/?{encoded_query}&sig={random.randint(1, 10000)}"
                    img_response = requests.get(unsplash_url, headers=headers, timeout=10)
                    
                    if img_response.status_code == 200:
                        try:
                            img = Image.open(BytesIO(img_response.content))
                            
                            file_extension = img.format.lower() if img.format else "jpg"
                            img_path = os.path.join(output_dir, f"{search_term.replace(' ', '_')}_{downloaded_count}.{file_extension}")
                            img.save(img_path)
                            
                            print(f"Downloaded from fallback: {img_path}")
                            downloaded_count += 1
                            
                            # Add a delay
                            time.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))
                            
                        except Exception as e:
                            print(f"Error with fallback image: {e}")
            except Exception as e:
                print(f"Fallback download failed: {e}")
        
        print(f"Successfully downloaded {downloaded_count} images")
        return downloaded_count
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return 0

def selenium_scraper(search_term, output_dir, num_images=None):
    """
    Alternative scraper using Selenium - only used if installed
    """
    if num_images is None:
        num_images = config.DEFAULT_NUM_IMAGES
        
    print(f"Scraping with Selenium for '{search_term}'...")
    
    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to import Selenium components with error handling
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("Selenium not installed. Falling back to regular scraper.")
        return scrape_images(search_term, output_dir, num_images)
    
    # Setup Chrome options with robust error handling
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(f"user-agent={config.USER_AGENT}")
        
        # Try to find a ChromeDriver executable if available
        chromedriver_paths = [
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            os.path.expanduser("~/chromedriver"),
        ]
        
        driver = None
        for driver_path in chromedriver_paths:
            if os.path.exists(driver_path):
                try:
                    service = Service(executable_path=driver_path)
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    print(f"Successfully initialized Chrome with driver at {driver_path}")
                    break
                except Exception as e:
                    print(f"Failed to initialize Chrome with driver at {driver_path}: {e}")
                    continue
        
        # If we couldn't find a valid driver path, try the default location
        if driver is None:
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                print(f"Failed to initialize Chrome with default driver: {e}")
                print("Falling back to regular scraper")
                return scrape_images(search_term, output_dir, num_images)
        
        # Format the search URL
        encoded_query = urllib.parse.quote(search_term)
        search_url = f"https://yandex.com/images/search?text={encoded_query}"
        
        # Add adult content parameter if needed
        if any(term in search_term.lower() for term in ['nsfw', 'adult', 'xxx']):
            search_url += "&noreask=1&adult=1"
            
        print(f"Opening URL: {search_url}")
        try:
            driver.get(search_url)
        except Exception as e:
            print(f"Failed to open URL: {e}")
            driver.quit()
            print("Falling back to regular scraper")
            return scrape_images(search_term, output_dir, num_images)
        
        # Wait for images to load with timeout handling
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".serp-item__thumb"))
            )
        except Exception as e:
            print(f"Timed out waiting for images to load: {e}")
            # Save page source for debugging
            with open(os.path.join(output_dir, "selenium_page_source.html"), 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print("Saved page source for debugging")
            driver.quit()
            print("Falling back to regular scraper")
            return scrape_images(search_term, output_dir, num_images)
        
        # Scroll down to load more images
        try:
            for _ in range(min(num_images // 10 + 1, 5)):  # Scroll a few times
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
        except Exception as e:
            print(f"Error during scrolling: {e}")
            # Continue anyway, we might have loaded some images
        
        # Find all image elements
        try:
            img_elements = driver.find_elements(By.CSS_SELECTOR, ".serp-item__thumb")
            print(f"Found {len(img_elements)} image thumbnails")
        except Exception as e:
            print(f"Error finding images: {e}")
            driver.quit()
            print("Falling back to regular scraper")
            return scrape_images(search_term, output_dir, num_images)
        
        # If we didn't find any images, try the fallback
        if not img_elements:
            print("No images found with Selenium, falling back to regular scraper")
            driver.quit()
            return scrape_images(search_term, output_dir, num_images)
        
        # Get image URLs by clicking each thumbnail
        image_urls = []
        image_hashes = set()
        downloaded_count = 0
        
        for img in img_elements[:num_images*2]:  # Check more than we need in case some fail
            if downloaded_count >= num_images:
                break
                
            try:
                # Click on the thumbnail to get the full image
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", img)
                time.sleep(0.5)  # Wait for scrolling to finish
                driver.execute_script("arguments[0].click();", img)
                time.sleep(1)
                
                # Wait for the full-size image container
                try:
                    full_img = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".MMImage-Origin"))
                    )
                    img_url = full_img.get_attribute("src")
                except:
                    # Try alternate selectors if the first one failed
                    for selector in [".MMImage-Preview", "img.MMImage-Origin", ".MMImage"]:
                        try:
                            full_img = driver.find_element(By.CSS_SELECTOR, selector)
                            img_url = full_img.get_attribute("src")
                            if img_url and img_url.startswith("http"):
                                break
                        except:
                            img_url = None
                
                if img_url and img_url.startswith("http"):
                    print(f"Found image URL: {img_url}")
                    
                    # Download the image
                    headers = {"User-Agent": config.USER_AGENT}
                    try:
                        img_response = requests.get(img_url, headers=headers, timeout=10)
                        
                        if img_response.status_code == 200:
                            # Process the image
                            img_data = Image.open(BytesIO(img_response.content))
                            
                            # Handle transparency
                            if img_data.mode == 'P' and 'transparency' in img_data.info:
                                img_data = img_data.convert('RGBA')
                                
                            # Check for duplicates
                            img_hash = get_image_hash(img_data)
                            if img_hash in image_hashes:
                                print("Skipping duplicate image")
                            else:
                                image_hashes.add(img_hash)
                                
                                # Save the image
                                file_extension = img_data.format.lower() if img_data.format else "jpg"
                                img_path = os.path.join(output_dir, f"{search_term.replace(' ', '_')}_{downloaded_count}.{file_extension}")
                                img_data.save(img_path)
                                
                                print(f"Downloaded: {img_path}")
                                downloaded_count += 1
                    except Exception as e:
                        print(f"Error downloading image: {e}")
                
                # Close the preview using multiple methods for robustness
                try:
                    close_btn = driver.find_element(By.CSS_SELECTOR, ".MMViewerModal-Close")
                    close_btn.click()
                    time.sleep(0.5)
                except:
                    try:
                        # If we can't find the close button, try pressing ESC
                        from selenium.webdriver.common.keys import Keys
                        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        time.sleep(0.5)
                    except:
                        try:
                            # Or try clicking outside the modal
                            outside_el = driver.find_element(By.TAG_NAME, "body")
                            outside_el.click()
                            time.sleep(0.5)
                        except:
                            # As a last resort, reload the page
                            driver.get(search_url)
                            time.sleep(2)
                
            except Exception as e:
                print(f"Error with image: {e}")
                # Try to reset the browser state and continue with next image
                try:
                    driver.get(search_url)
                    time.sleep(2)
                except:
                    pass
        
        driver.quit()
        print(f"Successfully downloaded {downloaded_count} images")
        
        # If we didn't get enough images, supplement with the regular scraper
        if downloaded_count < num_images:
            print(f"Only got {downloaded_count}/{num_images} images with Selenium, supplementing with regular scraper")
            more_images = scrape_images(search_term, output_dir, num_images - downloaded_count)
            return downloaded_count + more_images
        
        return downloaded_count
        
    except Exception as e:
        print(f"Selenium error: {e}")
        traceback.print_exc()
        # Fall back to regular scraper
        print("Selenium failed completely, falling back to regular scraper")
        return scrape_images(search_term, output_dir, num_images)

# Choose the appropriate scraper based on available dependencies
def get_best_scraper():
    """Determine the best available scraper based on installed dependencies"""
    # For now, use the regular scraper by default
    print("Using regular scraper as the default")
    return scrape_images
    
    # Uncomment this when Selenium issues are resolved
    # try:
    #     import selenium
    #     print("Using Selenium-based scraper")
    #     return selenium_scraper
    # except ImportError:
    #     print("Selenium not available, using standard scraper")
    #     return scrape_images

# Main function that decides which scraper to use
def scrape_images_auto(search_term, output_dir, num_images=None):
    """
    Main scraping function that chooses the appropriate scraper
    """
    scraper_func = get_best_scraper()
    return scraper_func(search_term, output_dir, num_images)