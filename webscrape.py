import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json

# List of account URLs
accounts = [
    "https://x.com/elonmusk/highlights",
    "https://x.com/ayeejuju/highlights",
    "https://x.com/RockingQAngel/highlights",
    "https://x.com/unknown_th82411/highlights"
]

# # Set up the browser
driver = webdriver.Chrome()

# Output folder for images
output_folder = "Webscrape_result/images"
os.makedirs(output_folder, exist_ok=True)

# JSON data structure
data = {}

for account in accounts:

    print(f"Processing account: {account}")
    driver.get(account)
    time.sleep(5)  # Wait for the page to load

    post_count = 0

    # Scroll and extract data until 5 posts are found
    scroll_attempts = 0  # To limit infinite scrolling
    max_attempts = 20    # Maximum number of scrolls allowed

    while post_count < 5 and scroll_attempts < max_attempts:  # Limit to 5 posts per account
        # Extract post elements
        post_elements = driver.find_elements(By.XPATH, "//article")

        for post in post_elements:
            if post_count >= 5:
                break  # Stop if we already have 10 posts

            try:
                # Extract URL
                url_element = post.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
                url = url_element.get_attribute("href")
                post_id = url.split("/")[-1]  # Extract the post ID from the URL

                # Extract Text
                try:
                    text_element = post.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
                    text = text_element.text
                except:
                    text = ""

                # Check for Images (exclude videos)
                image_elements = post.find_elements(By.XPATH, ".//img[contains(@src, 'twimg.com/media')]")
                video_elements = post.find_elements(By.XPATH, ".//video")

                # Filter posts with text and images but no videos
                if text and image_elements and not video_elements:
                    # Get the first image URL
                    img_url = image_elements[0].get_attribute("src")

                    # Add to JSON data
                    data[post_id] = {
                        "img_url": img_url,
                        "labels": [-1],
                        "tweet_url": url,
                        "tweet_text": text,
                        "labels_str": ["Empty"]
                    }

                    # Download and save the image
                    image_path = os.path.join(output_folder, f"{post_id}.jpg")
                    response = requests.get(img_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, "wb") as file:
                            for chunk in response.iter_content(1024):
                                file.write(chunk)
                        print(f"Saved Image: {image_path}")

                    print(f"Processed Post ID: {post_id}")
                    post_count += 1

            except Exception as e:
                # Skip if any element is missing or not matching criteria
                print(f"Error processing post: {e}")

        # Scroll down to load more posts
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(8)  # Adjust sleep time as needed
        scroll_attempts += 1




# Save JSON data to file
output_json_file = os.path.join("Webscrape_result/Webscrape_GT.json")
with open(output_json_file, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"JSON data saved to {output_json_file}")
driver.quit()
