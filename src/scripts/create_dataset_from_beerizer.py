from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup

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

beer_sections = BeautifulSoup(driver.page_source, 'html.parser').find_all('div', class_='beer-inner-top')

for beer_section in beer_sections:
    # get the name of the beer
    title_span = beer_section.find('span', class_='title', attrs={'itemprop': 'name'})

    if title_span:
        strong_text = title_span.find('strong')
        print(strong_text.text.strip())

    price_span = beer_section.find('span', class_='price')
    if price_span:
        price = price_span.text.strip()
        print(price)
