from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

base_url = "https://www.trustpilot.com/review/www.kfc.com"

reviews_data = []
page = 1


while True:
    url = f"{base_url}?page={page}"
    print(f"Scraping page {page}: {url}")

    driver.get(url)

    try:
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article")))
    except:
        print("No more reviews found.")
        break

    reviews = driver.find_elements(By.CSS_SELECTOR, "article")

    # if empty → stop
    if len(reviews) == 0:
        print("Reached last page.")
        break

    for review in reviews:
        def safe_find(by, value, attr=None):
            try:
                el = review.find_element(by, value)
                return el.get_attribute(attr) if attr else el.text
            except:
                return "N/A"

        name = safe_find(By.CSS_SELECTOR, "span[data-consumer-name-typography]")
        rating = safe_find(By.CSS_SELECTOR, "div[data-service-review-rating]", "data-service-review-rating")
        title = safe_find(By.CSS_SELECTOR, "h2")
        content = safe_find(By.CSS_SELECTOR, "p")
        date = safe_find(By.CSS_SELECTOR, "time", "datetime")

        reviews_data.append({
            "Name": name,
            "Rating": rating,
            "Title": title,
            "Review": content,
            
        })

    page += 1

driver.quit()


df = pd.DataFrame(reviews_data)
df.to_csv("kfc_trustpilot_all_reviews.csv", index=False)

print(f"Done! Scraped {len(df)} reviews.")