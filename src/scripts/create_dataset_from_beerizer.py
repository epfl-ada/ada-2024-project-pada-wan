from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import sys
import numpy as np

load_dotenv()
url = 'https://beerizer.com/?page1'
path_to_firefox = os.getenv('PATH_TO_FIREFOX')

options = Options()
options.headless = True
options.binary_location = os.path.join(path_to_firefox, 'firefox')
options.add_argument('--headless')

driver = webdriver.Firefox(service=Service(os.path.join(path_to_firefox, 'geckodriver')), options=options)

driver.get(url)



time.sleep(0.1)


def get_largest_page(page_source):
    max_page = 0
    for page_switch_el in page_source.find_all('a', class_='dynamic-url scroll-to-beers exact-link'):
        try:
            largest_page = int(page_switch_el.get_text(strip=True)) > max_page
        except ValueError:
            continue
        if largest_page:
            max_page = int(page_switch_el.get_text(strip=True))
    return max_page


try:
    page_source = BeautifulSoup(driver.page_source, 'html.parser')

    beer_sections = page_source.find_all('div', class_='beer-inner-top')

    print("number of pages ",get_largest_page(page_source))
    for beer_section in beer_sections:
        # get the name of the beer
        title_span = beer_section.find('span', class_='title', attrs={'itemprop': 'name'})
        if title_span:
            strong_text = title_span.find('strong')
            print(strong_text.text.strip())

        brewery_span = beer_section.find('span', class_='brewery-title')
        if brewery_span:
            img = brewery_span.find('img')
            if img:
                print(img.get('alt'))
                img.extract()
            print(brewery_span.get_text(strip=True))

        price_span = beer_section.find('span', class_='price')
        if price_span:
            price = price_span.text.strip()
            print(price)

        pack_info_div = beer_section.find('div', class_='pack-info')
        if pack_info_div:
            span_pack_info_div = pack_info_div.find('span')
            if span_pack_info_div:
                print(span_pack_info_div.get_text(strip=True))
                print(span_pack_info_div.get('title'))

        additional_info = beer_section.find('div', class_='right-item-row rating-abv-rpc')
        if additional_info:
            untapped_rating = additional_info.find('a', class_='untappd untappd-mouseover')
            if untapped_rating:
                print(untapped_rating.get_text(strip=True))

            percentage_alcohol = additional_info.find('span', class_='abv value')
            if percentage_alcohol:
                print(percentage_alcohol.get_text(strip=True))
        elif beer_section.find('div', class_='bundle-header'):
                continue


        beer_type_div = beer_section.find('div', class_='right-item-row style')
        if beer_type_div:
            print(beer_type_div.get_text(strip=True))

        print()
finally:
    driver.quit()

sys.exit()