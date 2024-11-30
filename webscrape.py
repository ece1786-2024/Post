import json
import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By


class XScraper:
    def __init__(self, accounts, output_folder="Webscrape_result/images"):
        self.accounts = accounts
        self.output_folder = output_folder
        self.driver = webdriver.Chrome()
        self.data = {}

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def scrape_account(self, account_url, max_posts=5, max_scroll_attempts=20):
        """
        Scrape posts from a given Twitter account URL.

        :param account_url: URL of the Twitter account highlights page.
        :param max_posts: Maximum number of posts to scrape per account.
        :param max_scroll_attempts: Maximum scroll attempts to prevent infinite loops.
        """
        print(f"Processing account: {account_url}")
        self.driver.get(account_url)
        time.sleep(5)  # Allow the page to load

        post_count = 0
        scroll_attempts = 0

        while post_count < max_posts and scroll_attempts < max_scroll_attempts:
            post_elements = self.driver.find_elements(By.XPATH, "//article")
            for post in post_elements:
                if post_count >= max_posts:
                    break

                try:
                    # Extract data from post
                    post_data = self.extract_post_data(post)
                    if post_data:
                        post_id = post_data["post_id"]
                        self.data[post_id] = post_data

                        # Download image
                        self.download_image(post_data["img_url"], post_id)

                        print(f"Processed Post ID: {post_id}")
                        post_count += 1
                except Exception as e:
                    print(f"Error processing post: {e}")

            # Scroll down to load more posts
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(8)
            scroll_attempts += 1

    def extract_post_data(self, post):
        """
        Extract data from a post element.

        :param post: WebElement representing a Twitter post.
        :return: A dictionary containing post data or None if the post doesn't meet criteria.
        """
        try:
            url_element = post.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
            url = url_element.get_attribute("href")
            post_id = url.split("/")[-1]

            text_element = post.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
            text = text_element.text

            image_elements = post.find_elements(By.XPATH, ".//img[contains(@src, 'twimg.com/media')]")
            video_elements = post.find_elements(By.XPATH, ".//video")

            if text and image_elements and not video_elements:
                img_url = image_elements[0].get_attribute("src")
                return {
                    "post_id": post_id,
                    "img_url": img_url,
                    "labels": [-1],
                    "tweet_url": url,
                    "tweet_text": text,
                    "labels_str": ["Empty"],
                }
        except Exception as e:
            print(f"Error extracting post data: {e}")
        return None

    def download_image(self, img_url, post_id):
        """
        Download an image from a URL and save it to the output folder.

        :param img_url: URL of the image.
        :param post_id: ID of the post for naming the image file.
        """
        try:
            image_path = os.path.join(self.output_folder, f"{post_id}.jpg")
            response = requests.get(img_url, stream=True)
            if response.status_code == 200:
                with open(image_path, "wb") as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print(f"Saved Image: {image_path}")
        except Exception as e:
            print(f"Error downloading image: {e}")

    def save_to_json(self, output_file="Webscrape_result/Webscrape_GT.json"):
        """
        Save scraped data to a JSON file.

        :param output_file: Path to the output JSON file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as json_file:
            json.dump(self.data, json_file, indent=4)
        print(f"JSON data saved to {output_file}")

    def close(self):
        """Close the browser."""
        self.driver.quit()

    def run(self):
        """
        Run the scraper for all accounts.
        """
        try:
            for account in self.accounts:
                self.scrape_account(account)
            self.save_to_json()
        finally:
            self.close()


if __name__ == "__main__":
    accounts = [
        "https://x.com/elonmusk/highlights",
        "https://x.com/ayeejuju/highlights",
        "https://x.com/RockingQAngel/highlights",
        "https://x.com/unknown_th82411/highlights",
    ]

    scraper = XScraper(accounts)
    scraper.run()
