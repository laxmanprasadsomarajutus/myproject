import os
import re
import time
import json
import colorlog
import requests
import logging
import pandas as pd
import lxml
import amazon_scrape_toolkit as ast
import bs4

log_file = "./make_dataset-v2.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

color_stdout = logging.StreamHandler()
color_stdout.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "blue",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

# Add the handlers to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(color_stdout)
logger.setLevel(logging.INFO)

HEADERS = ast.AmazonHeaders(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "en-US",
)


def fetch_webpage(url):
    response = session.get(url, headers=HEADERS.req)
    response.raise_for_status()
    return response.content


if not os.path.exists("../../data/raw-v2.json"):
    with open("../../data/raw-v2.json", "w", newline="") as csvfile:
        pass

try:
    existing = pd.read_json("../data/raw.json")
    existing.set_index(keys=["product_id"], inplace=True)
    existing_indices = list(set(existing.index.values.tolist()))
except:
    existing_indices = []

start_time = time.monotonic()
fetch_times = []

scraped_phones = set()
all_phones = []

session = requests.Session()
failed = failed_reasons = []
scraped_ids = []
data = []


@ast.product_scraper()
def extract_product_info(soup, product_id):
    product_info = {"product_id": product_id}

    compare_table = soup.find("table", {"id": "HLCXComparisonTable"})

    mrp = float(soup.find("span", class_="a-price-whole").text.strip().replace(",", ""))
    model_name = str(soup.find("span", {"id": "productTitle"}).text.strip())

    product_info["mrp"] = mrp
    product_info["model_name"] = model_name.replace("|", "(").split("(")[0]

    if "(renewed)" in model_name.lower():
        logger.warn(f"PRODUCT IS RENEWED TYPE {product_id}")
        return

    if product_id in existing_indices:
        logger.warn(f"PRODUCT IS EXISTING TYPE {product_id}")
        return

    # --------- MINI INFO TABLE

    mini_table = {}
    table = soup.find("table", {"class": "a-normal a-spacing-micro"})

    if isinstance(table, bs4.Tag):
        for row in table.find_all("tr"):
            key = row.find("td", {"class": "a-span3"}).text.strip()
            value = row.find("td", {"class": "a-span9"}).text.strip()

            if key == "Model Name":
                product_info["model_name"] = value
                continue

            mini_table[key.lower()] = value

    # --------- PRODUCT SPECS TABLE

    product_spec_table = {}
    table = soup.find("table", {"id": "productDetails_techSpec_section_1"})
    if isinstance(table, bs4.Tag):
        for row in table.find_all("tr"):
            key = row.find("th").text.strip()
            value = row.find("td").text.strip()
            value = value.replace("\u200e", "")
            product_spec_table[key.lower()] = value

    # --------- COMPARE TABLE PHONE INFO

    compare_table_info = {}
    phone_info_rows = (
        compare_table.find_all("tr")[6:] if isinstance(compare_table, bs4.Tag) else []
    )
    for row in phone_info_rows:
        assert isinstance(row, bs4.Tag)
        property_value = row.find("td")

        if property_value is None:
            continue

        property_tag = row.find("th").find("span")
        assert isinstance(property_tag, bs4.Tag)
        property_name = property_tag.text.strip()
        property_value = property_value.text.strip()
        compare_table_info[property_name.lower()] = property_value

    # --------- EXTRACTING DATA FROM PAGE

    # so far we have mrp and model_name
    # and containers of mini_table, product_spec_table, compare_table_info
    logger.debug(f"MINI TABLE KEYS: {list(mini_table.keys())}")
    logger.debug(f"PRODUCT SPEC TABLE KEYS: {list(product_spec_table.keys())}")
    logger.debug(f"COMPARE TABLE KEYS: {list(compare_table_info.keys())}")

    def search_nearest_column(fnc, msg=None):
        cti, pst, mt = compare_table_info, product_spec_table, mini_table
        for dictionary in (cti, pst, mt):
            keys = dictionary.keys()
            for key in keys:
                if not fnc(key):
                    continue

                return dictionary[key]

        assert not isinstance(msg, str), msg
        return

    os = search_nearest_column(
        lambda key: key == "os"
        or key.find("operating") > -1
        and key.find("system") > -1,
        "OS was not Found",
    )

    ram = search_nearest_column(
        lambda key: key.find("ram") > -1,
        "RAM was not Found",
    )

    dimensions = (
        search_nearest_column(
            lambda key: key.find("dimensions") > -1,
            "product dimensions was not Found",
        )
        .split(";")[0]
        .strip()
    )

    weight = (
        compare_table_info.get("item weight")
        if compare_table_info.get("item weight") is not None
        else product_spec_table.get("item weight")
    )

    if weight is None:
        weight = search_nearest_column(
            lambda key: key.find("dimensions") > -1,
            "product dimensions was not Found",
        )

    assert isinstance(weight, str) and (
        "g" in weight or "k" in weight
    ), "Weight isnt right smh"

    weight = weight.split(";")[-1]
    logging.debug(f"Weight: {weight}")

    inbuilt_storage = search_nearest_column(
        lambda key: (
            key.find("storage") > -1
            or key.find("inbuilt") > -1
            or key.find("memory") > -1
        ),
        "Inbuilt Storage Amount Wasnt Found",
    )

    battery_power = search_nearest_column(
        lambda key: (key.find("battery") > -1 and key.find("power") > -1),
        "Battery Power Wasnt Found",
    )

    battery_type = search_nearest_column(
        lambda key: (
            key.find("batteries") > -1
            or (key.find("battery") > -1 and key.find("type") > -1)
        ),
        # "Battery Type Wasnt Found", # instead i want to use fillna here if not Found
    )

    camera = search_nearest_column(
        lambda key: key.find("camera") > -1,
        "Camera Details Wasnt Found",
    )

    warranty = search_nearest_column(
        lambda key: (key.find("warranty") > -1),
    )

    form_factor = product_spec_table.get("form factor")  # can delete but wont
    # assert form_factor is not None, "Form Factor wasn't Found"

    manufacturer = product_spec_table.get("manufacturer")
    assert manufacturer is not None, "Manufacturer wasn't Found"

    # --------- SAVING DATA TO PRODUCT_INFO

    product_info["os"] = os
    product_info["ram"] = ram
    product_info["inbuilt_storage"] = inbuilt_storage
    product_info["dimensions"] = dimensions
    product_info["weight"] = weight
    product_info["battery_power"] = battery_power
    product_info["battery_type"] = battery_type
    product_info["camera"] = camera
    product_info["warranty"] = warranty
    product_info["form_factor"] = form_factor
    product_info["manufacturer"] = manufacturer

    # screen type

    return product_info


product_ids_to_scrape = ast.get_all_product_ids(
    "https://www.amazon.in/s?rh=n%3A1805560031&fs=true&ref=lp_1805560031_sar",
    HEADERS,
    pages_to_scrape=50
)


try:
    while len(product_ids_to_scrape) > 0 and len(scraped_phones) < 1000:
        id_to_scrape = product_ids_to_scrape.pop()
        logging.info(f"Scraping product with ID: {id_to_scrape}")

        if id_to_scrape in scraped_phones:
            continue

        try:
            link = f"https://www.amazon.in/dp/{id_to_scrape}"
            soup = bs4.BeautifulSoup(fetch_webpage(link), "lxml")
            output: ast.ProductInfo = extract_product_info(soup, id_to_scrape)
        except AssertionError as e:
            logger.info(f"PRODUCT SCRAPING FAILED: [{id_to_scrape}]")
            logger.warning(f"REASON: {str(e)}")
            failed.append(id_to_scrape)
            failed_reasons.append(str(e))
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for product {id_to_scrape}: {e}")
            failed_reasons.append(str(e))
            failed.append(id_to_scrape)
            continue

        if output is None:
            continue

        scraped_phones.add(id_to_scrape)

        product_data = dict()
        product_data.update(output.data)

        if output.other_products is not None:
            product_ids_to_scrape.update(output.other_products)

        if output.ratings is not None:
            product_data.update(output.ratings)

        if product_data not in data:
            logging.info(f"product_scraped and saved [{id_to_scrape}]")
            data.append(product_data)
        else:
            logging.info(f"product data not new and so not saved [{id_to_scrape}]")

        logging.info(f"Remaining products to scrape: {len(product_ids_to_scrape)}")

except KeyboardInterrupt:
    pass

with open("../../data/raw-v2.json", "r") as file:
    try:
        data = json.load(file)
    except:
        data = []

    current_no_of_rows = len(data)

# Append a variable containing a list of objects
finalized_new_phones = [
    phone
    for phone in all_phones
    if isinstance(phone, dict)
    and phone["product_id"] not in existing_indices
    and phone not in data
]
data.extend(finalized_new_phones)
new_no_of_rows = len(data)

# Save the updated data back to the original file
with open("../../data/raw-v2.json", "w") as file:
    json.dump(data, file, indent=4)

failed_reasons = list(filter(lambda x: isinstance(x, str), failed_reasons))

logger.info("Total Time: %s", time.monotonic() - start_time)
logger.info("TIME FOR FETCHING: %s", fetch_times)
logger.info(
    "AVG TIME FOR FETCHING: %s",
    sum(fetch_times) / len(fetch_times) if len(fetch_times) > 0 else "NA",
)
logger.info(
    "NEWLY ADDED %s ROWS",
    len(finalized_new_phones),
)
logger.info("FAILED %s ROWS", len(failed))
logger.info("OLD NO OF ROWS: %s", current_no_of_rows)
logger.info("NEW NO OF ROWS: %s", new_no_of_rows)
logger.info("FAILED REASONS: %s", failed_reasons)
logger.info(
    "MAJOR FAILED REASON: %s",
    [
        (value, failed_reasons.count(value))
        for value in sorted(
            list(set(failed_reasons)),
            key=lambda reason: failed_reasons.count(reason),
            reverse=True,
        )
    ],
)
