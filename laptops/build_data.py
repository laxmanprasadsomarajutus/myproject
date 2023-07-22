import os
import time
import json
from amazon_scrape_toolkit.main import ProductInfo
import colorlog
import requests
import logging
import pandas as pd
import amazon_scrape_toolkit as ast
import bs4

log_file = "./debug.log"
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
color_stdout.setLevel(logging.INFO)

# Add the handlers to the logger
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.addHandler(color_stdout)
logger.setLevel(logging.DEBUG)

HEADERS = ast.AmazonHeaders(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "en-US",
)

DATA_PATH = "./data/raw.json"


if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, "w", newline="") as csvfile:
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


def fetch_webpage(url):
    response = session.get(url, headers=HEADERS.req)
    response.raise_for_status()
    return response.content


@ast.product_scraper(should_raise=True)
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

    # --------- EXTRACTING DATA FROM PAGE AND SAVING IT

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

    product_info["screen_size"] = search_nearest_column(
        lambda key: (
            (key.find("screen") > -1 or key.find("display") > -1)
            and key.find("size") > -1
        ),
        "screen_size was not found",
    )

    product_info["display_resolution"] = search_nearest_column(
        lambda key: key.find("resolution") > -1,
        # "display_resolution wasnt found",
    )

    product_info["os"] = search_nearest_column(
        lambda key: key == "os"
        or key.find("operating") > -1
        and key.find("system") > -1,
        "OS was not Found",
    )

    product_info["hard_disk_type"] = search_nearest_column(
        lambda key: key.find("hard") > -1
        and key.find("disk") > -1
        and (key.find("description") > -1 or key.find("technology") > -1),
        # "hard disk drive technology not found",
    )

    if product_info["hard_disk_type"] is None:
        product_info["hard_disk_type"] = "SSD"

    product_info["hard_drive_size"] = search_nearest_column(
        lambda key: (
            (key.find("hard") > -1 and key.find("drive") > -1)
            or (key.find("flash") > -1 and key.find("memory"))
        )
        and key.find("size") > -1,
        "hard disk drive size not found",
    )

    ram_memory = search_nearest_column(
        lambda key: (
            key.find("ram") > -1
            and key.find("memory") > -1
            and key.find("installed") > -1
            and key.find("size") > -1
        )
    )

    if ram_memory is None:
        ram_memory = search_nearest_column(
            lambda key: (
                key.find("ram") > -1
                and key.find("memory") > -1
                and key.find("max") > -1
                and key.find("size") > -1
            )
        )

    assert ram_memory is not None, "RAM amount wasnt found"
    product_info["ram_memory"] = ram_memory

    processor_brand = product_spec_table.get("processor brand")
    assert processor_brand is not None, "Processor Brand Not Found"
    product_info["processor_brand"] = processor_brand

    product_info["processor_name"] = search_nearest_column(
        lambda key: (
            key.find("processor") > -1
            and (key.find("name") > -1 or key.find("type") > -1)
        ),
        "Processor Name Not Found",
    )

    processor_speed = product_spec_table.get("processor speed")
    processor_count = product_spec_table.get("processor count")
    assert processor_speed and processor_count, "Process Speed or Count not Found"

    product_info["processor_speed"] = processor_speed
    product_info["processor_count"] = processor_count

    product_info["display_type"] = product_spec_table.get("display type")

    product_info["product_dimensions"] = search_nearest_column(
        lambda key: key.find("dimensions") > -1
        and (key.find("product") > -1 or key.find("package") > -1),
        "Product Dimensions Doesnt Exist!!!!",
    )

    batteries = product_spec_table.get("batteries")
    assert batteries is not None, "Batteries not Found"
    product_info["batteries"] = batteries

    form_factor = product_spec_table.get("form factor")
    assert form_factor is not None, "Form Factor wasn't Found"
    product_info["form_factor"] = form_factor

    product_info["audio_details"] = search_nearest_column(
        lambda key: key.find("audio") > -1,
        # "Audio Not Found",
    )

    product_info["speaker_details"] = search_nearest_column(
        lambda key: key.find("speaker") > -1,
        # "Speaker Details Not Found",
    )

    product_info["connector_types"] = search_nearest_column(
        lambda key: (key.find("connectivity") > -1 or key.find("connector"))
        and key.find("type") > -1,
        "Connector Type Not Found",
    )

    product_info["graphics chipset"] = search_nearest_column(
        lambda key: key == "graphics chipset brand"
    )

    product_info["graphics type"] = search_nearest_column(
        lambda key: key.find("graphics card") > -1
        and (key.find("description") > -1 or key.find("interface") > -1)
    )

    product_info["graphics ram type"] = search_nearest_column(
        lambda key: key == "graphics ram type"
    )

    product_info["graphics details"] = search_nearest_column(
        lambda key: key == "graphics coprocessor"
    )

    assert product_spec_table.get("brand") is not None, "Brand not Specified"
    product_info["brand"] = product_spec_table.get("brand")

    return product_info


product_ids_to_scrape = ast.get_all_product_ids(
    "https://www.amazon.in/s?rh=n%3A1375424031&fs=true&ref=lp_1375424031_sar",
    HEADERS,
)


try:
    i = 1
    while len(product_ids_to_scrape) > 0 and len(all_phones) < 10000:
        id_to_scrape = product_ids_to_scrape.pop()
        logging.info(f"Scraping product {i} with ID: {id_to_scrape}")
        i += 1

        if id_to_scrape in scraped_phones:
            continue

        try:
            link = f"https://www.amazon.in/dp/{id_to_scrape}"
            soup = bs4.BeautifulSoup(fetch_webpage(link), "lxml")
            output: ast.ProductInfo = extract_product_info(soup, id_to_scrape)
        except (AssertionError, AttributeError) as e:
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

        if product_data not in all_phones:
            logging.info(f"product_scraped and saved [{id_to_scrape}]")
            all_phones.append(product_data)
        else:
            logging.error(f"product data not new and so not saved [{id_to_scrape}]")

        logging.info(f"Remaining products to scrape: {len(product_ids_to_scrape)}")
        logging.info(f"Scraped {len(all_phones)} Products")
        logging.info("--------")

except KeyboardInterrupt:
    pass

with open(DATA_PATH, "r") as file:
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
with open(DATA_PATH, "w") as file:
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
