import re, sys
import time, json
import requests
import logging
import pandas as pd
import lxml
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import bs4

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
BASE_URL = (
    "https://www.amazon.in/Smartphones/b/ref=dp_bc_aui_C_4?ie=UTF8&node=1805560031"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US",
}
MAX_RETRIES = 10

chrome_options = Options()
chrome_options.add_argument("--headless")
capabilities = DesiredCapabilities.CHROME
capabilities["pageLoadStrategy"] = "eager"

phones_id = []
phones_link = []

with webdriver.Chrome(
    options=chrome_options, desired_capabilities=capabilities
) as driver:
    driver.get(BASE_URL)
    driver.refresh()

    driver.implicitly_wait(2)

    uls = driver.find_elements(
        By.CSS_SELECTOR,
        "ul.a-unordered-list.a-nostyle.a-horizontal.octopus-pc-card-list.octopus-pc-card-height-v3",
    )[:-1]

    for ul in uls:
        span_tags = ul.find_elements(By.CSS_SELECTOR, "li span.a-list-item")
        a_tag_links = [
            tag.find_element(By.TAG_NAME, "a").get_attribute("href")
            for tag in span_tags
        ]

        phone_id = []

        for link in a_tag_links:
            id = link.split("=")[-1]

            if id in span_tags:
                a_tag_links.remove(link)
                logging.info(f"found matching id: {id}")
                continue

            phone_id.append(id)

        phones_id.extend(phone_id)
        phones_link.extend(a_tag_links)

assert len(phones_id) == len(phones_link)
logging.info(len(phones_id))

existing = pd.read_json("../../data/raw-v1.json")
existing.set_index(keys=["product_id"], inplace=True)

start_time = time.monotonic()
fetch_times = []

scraped_phones = []
scraped_phones_id = set(existing.index.values)
all_phones = []

requests_session = requests.Session()
to_scrape = list(zip(phones_id, phones_link))
failed = list()

try:
    while len(to_scrape) > 0:
        id, link = to_scrape.pop()
        time_taken = time.monotonic()
        already_scraped = id in scraped_phones_id
        logging.info(f"-----------------------------------")
        logging.info(f"PRODUCT REMAINING: {len(to_scrape)}")
        logging.info(f"PRODUCT SCRAPED: {len(scraped_phones)}")

        webpage = requests_session.get(link, headers=HEADERS)
        soup = bs4.BeautifulSoup(webpage.content, "lxml")
        product_info = {"product_id": id}

        try:
            compare_table = soup.find("table", {"id": "HLCXComparisonTable"})
            assert isinstance(compare_table, bs4.Tag)

            new_phones = [
                a_tag["href"]
                for a_tag in compare_table.find(
                    "tr", class_="comparison_table_image_row border-none"
                ).find_all("a")
            ][1:]
            new_phones_to_add = []

            for phone in new_phones:
                match = re.search(r"/dp/([A-Z0-9]+)", phone)

                if match:
                    competitor_id = match.group(1)
                    new_phones_to_add.append(
                        (competitor_id, f"https://www.amazon.in{phone}")
                    )

            new_phones_to_add = list(
                filter(
                    lambda x: (x not in to_scrape) and (x[0] not in scraped_phones_id),
                    new_phones_to_add,
                )
            )

            if already_scraped:
                to_scrape.extend(new_phones_to_add)
                logging.info(f"FOUND %s NEW PHONES", len(new_phones_to_add))
                continue

            mrp = float(
                soup.find("span", class_="a-price-whole").text.strip().replace(",", "")
            )
            product_info["mrp"] = mrp

            table = soup.find("table", {"class": "a-normal a-spacing-micro"})
            assert isinstance(table, bs4.Tag)

            for row in table.find_all("tr"):
                key = row.find("td", {"class": "a-span3"}).text.strip()
                value = row.find("td", {"class": "a-span9"}).text.strip()
                product_info[key] = value

            product_identification = (product_info.get("Model Name"), mrp)
            if product_identification in scraped_phones:
                logging.info(f"PRODUCT SCRAPED ALREADY {id} - {product_identification}")
                continue

            table = soup.find("table", {"id": "productDetails_techSpec_section_1"})
            assert isinstance(table, bs4.Tag)

            for row in table.find_all("tr"):
                key = row.find("th").text.strip()
                value = row.find("td").text.strip()
                value = value.replace("\u200e", "")
                product_info[key] = value

            phone_info_rows = compare_table.find_all("tr")[6:]
            for row in phone_info_rows:
                assert isinstance(row, bs4.Tag)
                property_value = row.find("td")

                if property_value is None:
                    continue

                property_name = row.find("th").find("span").text.strip()
                property_value = property_value.text.strip()
                product_info[property_name] = property_value

            rating_histogram_div = soup.find(
                "div", {"id": "cm_cr_dp_d_rating_histogram"}
            )
            assert isinstance(rating_histogram_div, bs4.Tag)
            avg_rating_div = rating_histogram_div.find(
                "div",
                {"class": "a-fixed-left-grid AverageCustomerReviews a-spacing-small"},
            )
            assert isinstance(avg_rating_div, bs4.Tag)
            avg_rating_span = avg_rating_div.find(
                "span",
                {
                    "data-hook": "rating-out-of-text",
                    "class": "a-size-medium a-color-base",
                },
            )
            assert isinstance(avg_rating_span, bs4.Tag)

            avg_rating = float(avg_rating_span.text.strip().split()[0])

            no_of_ratings_div = rating_histogram_div.find(
                "div",
                {"class": "a-row a-spacing-medium averageStarRatingNumerical"},
            )
            assert isinstance(no_of_ratings_div, bs4.Tag)
            no_of_ratings_span = no_of_ratings_div.find("span")
            assert isinstance(no_of_ratings_span, bs4.Tag)
            no_of_ratings = int(
                no_of_ratings_span.text.strip().split()[0].replace(",", "")
            )

            table = rating_histogram_div.find(
                "table", {"class": "a-normal a-align-center a-spacing-base"}
            )
            assert isinstance(table, bs4.Tag)

            rows = table.find_all("tr")
            individual_star_ratings = [
                (
                    row.find_all("td")[0].text.strip(),
                    row.find_all("td")[2].text.strip(),
                )
                for row in rows
            ]
            stared_ratings = {
                f"no of {key}": int((int(value[:-1]) / 100) * no_of_ratings)
                for key, value in individual_star_ratings
            }

            for key, value in stared_ratings.items():
                product_info[key] = value

        except AssertionError:
            logging.info(f"SOURCE PARTS COULD NOT BE FOUND: {id}")
            failed.append((id, link))
            continue

        except AttributeError:
            logging.info(f"SOURCE doesn't CONTAIN NEEDED INFO: {id}")
            continue

        fetch_times.append(round(time.monotonic() - time_taken, 2))
        scraped_phones.append(product_identification)
        to_scrape.extend(new_phones_to_add)
        all_phones.append(product_info)
        scraped_phones_id.add(id)
        logging.info(f"DONE WITH {id} [{len(scraped_phones)}]")
        time.sleep(1)

except KeyboardInterrupt:
    pass

new = [row for row in all_phones if row["product_id"] not in existing.index]

with open("../../data/raw-v1.json", "r") as file:
    data = json.load(file)
    current_no_of_rows = len(data)

# Append a variable containing a list of objects
data.extend(new)
new_no_of_rows = len(data)

# Save the updated data back to the original file
with open("../../data/raw-v1.json", "w") as file:
    json.dump(data, file, indent=4)

logging.info("Total Time: %s", time.monotonic() - start_time)
logging.info("TIME FOR FETCHING: %s", fetch_times)
logging.info("AVG TIME FOR FETCHING: %s", sum(fetch_times) / len(fetch_times))
logging.info("NEWLY ADDED %s ROWS", len(new))
logging.info("FAILED %s ROWS", len(failed))
logging.info("OLD NO OF ROWS: %s", current_no_of_rows)
logging.info("NEW NO OF ROWS: %s", new_no_of_rows)
