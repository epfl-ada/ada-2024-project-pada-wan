from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import sys
import pandas as pd

load_dotenv()
url_prepage = 'https://beerizer.com/?page='
url = 'https://beerizer.com/'
path_to_firefox = os.getenv('PATH_TO_FIREFOX')

options = Options()
options.headless = True
options.binary_location = os.path.join(path_to_firefox, 'firefox')
options.add_argument('--headless')


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


driver = webdriver.Firefox(service=Service(os.path.join(path_to_firefox, 'geckodriver')), options=options)

driver.get(url)

time.sleep(0.1)

page_source = BeautifulSoup(driver.page_source, 'html.parser')

max_page = get_largest_page(page_source)

ultimate_df = pd.DataFrame(
    columns=['Beer_name', 'Price', 'Origin', 'Rating', 'Brewery', 'Percentage', 'Beer_type', 'Countenance',
             'Countenance_per_litre'])

try:

    for i in range(max_page - 10):
        print("on page ", i)
        beer_dico = {}

        driver.get(url_prepage + str(i))

        time.sleep(0.2)

        page_source = BeautifulSoup(driver.page_source, 'html.parser')

        beer_sections = page_source.find_all('div', class_='beer-inner-top')


        for beer_section in beer_sections:
            # get the name of the beer
            title_span = beer_section.find('span', class_='title', attrs={'itemprop': 'name'})
            if title_span:
                strong_text = title_span.find('strong')
                beer_dico['Beer_name'] = strong_text.text.strip()

            # brewery name and origin
            brewery_span = beer_section.find('span', class_='brewery-title')
            if brewery_span:
                img = brewery_span.find('img')
                if img:
                    beer_dico['Origin'] = img.get('alt')
                    img.extract()
                beer_dico['Brewery'] = brewery_span.get_text(strip=True)

            # price
            price_span = beer_section.find('span', class_='price')
            if price_span:
                price = price_span.text.strip()
                beer_dico['Price'] = price

            # untappd rating
            pack_info_div = beer_section.find('div', class_='pack-info')
            if pack_info_div:
                span_pack_info_div = pack_info_div.find('span')
                if span_pack_info_div:
                    beer_dico['Countenance'] = span_pack_info_div.get_text(strip=True)
                    beer_dico['Countenance_per_litre'] = span_pack_info_div.get('title')

            # alcohol percentage if bundle discard
            additional_info = beer_section.find('div', class_='right-item-row rating-abv-rpc')
            if additional_info:
                untapped_rating = additional_info.find('a', class_='untappd untappd-mouseover')
                if untapped_rating:
                    beer_dico['Rating'] = untapped_rating.get_text(strip=True)

                percentage_alcohol = additional_info.find('span', class_='abv value')
                if percentage_alcohol:
                    beer_dico['Percentage'] = percentage_alcohol.get_text(strip=True)
            elif beer_section.find('div', class_='bundle-header'):
                continue

            # beer type
            beer_type_div = beer_section.find('div', class_='right-item-row style')
            if beer_type_div:
                beer_dico['Beer_type'] = beer_type_div.get_text(strip=True)

            ultimate_df.loc[len(ultimate_df)] = beer_dico
finally:
    driver.quit()
    ultimate_df.to_csv('official_beerizer_dataset.csv', index=False)

sys.exit()
