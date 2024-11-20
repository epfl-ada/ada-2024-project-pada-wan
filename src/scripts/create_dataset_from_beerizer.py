from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
import os
import time
from dotenv import load_dotenv

load_dotenv()
url = 'https://beerizer.com/?page1'
path_to_firefox = os.getenv('PATH_TO_FIREFOX')

options = Options()
options.headless = True
options.binary_location = os.path.join(path_to_firefox, 'firefox')
options.add_argument('--headless')

driver = webdriver.Firefox(service=Service(os.path.join(path_to_firefox, 'geckodriver')), options=options)

driver.get(url)

time.sleep(3)

print(driver.page_source)
