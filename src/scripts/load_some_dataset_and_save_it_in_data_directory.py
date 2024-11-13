import pandas as pd
import re
import pickle
import numpy as np
import seaborn as sns
import os

DATA_PATH = './data/'
BEER_ADVOCATE_PATH = 'BeerAdvocate'
MATCHED_BEER_DATA = 'matched_beer_data'
RATE_BEER_DATA = 'RateBeer'
print(os.path.join(DATA_PATH, MATCHED_BEER_DATA))

beer_advocate_beers = pd.read_csv(os.path.join(DATA_PATH, BEER_ADVOCATE_PATH, 'beers.csv'))
beer_advocate_breweries = pd.read_csv(os.path.join(DATA_PATH, BEER_ADVOCATE_PATH, 'breweries.csv'))
beer_advocate_users = pd.read_csv(os.path.join(DATA_PATH, BEER_ADVOCATE_PATH, 'users.csv'))

matched_beer_data_beers = pd.read_csv(os.path.join(DATA_PATH, MATCHED_BEER_DATA, 'beers.csv'))
matched_beer_data_breweries = pd.read_csv(os.path.join(DATA_PATH, MATCHED_BEER_DATA, 'breweries.csv'))
matched_beer_data_ratings = pd.read_csv(os.path.join(DATA_PATH, MATCHED_BEER_DATA, 'ratings.csv'))
matched_beer_data_users = pd.read_csv(os.path.join(DATA_PATH, MATCHED_BEER_DATA, 'users.csv'))
matched_beer_data_users_approx = pd.read_csv(os.path.join(DATA_PATH, MATCHED_BEER_DATA, 'users_approx.csv'))

rate_beer_beers = pd.read_csv(os.path.join(DATA_PATH, RATE_BEER_DATA, 'beers.csv'))
rate_beer_breweries = pd.read_csv(os.path.join(DATA_PATH, RATE_BEER_DATA, 'breweries.csv'))
rate_beer_users = pd.read_csv(os.path.join(DATA_PATH, RATE_BEER_DATA, 'users.csv'))


def extract_full_data(file_path: str, fields: list[str] = None) -> pd.DataFrame:
    """
    :param file_path: a file_path for the .txt file you want to extract
    :param fields: a list of fields to extract from the .txt file, if None the function will find the fields for you
    :return: a df containing all extracted data from the dataframe
    """
    data_list = []
    dico = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # if fields is None try to find all the fields automatically
    if fields is None:
        fields_cpy = []
        for line in lines:
            if line.strip() == '':
                continue
            field = re.match(r'^[^:]+', line).group(0)
            if field in fields_cpy:
                break
            fields_cpy.append(field)
    else:
        fields_cpy = fields

    for line in lines:
        if line.strip() == '':
            continue

        # check if the line starts with the first field and if that is so and the dico is not empty then start new dico
        if line.startswith(f"{fields_cpy[0]}:") and dico:
            data_list.append(dico)
            dico = {}

        for field in fields_cpy:
            if line.startswith(f"{field}:"):
                dico[field] = line[len(field) + 2:].strip()

    return pd.DataFrame(data_list)


ratings_ba_df = extract_full_data('./data/matched_beer_data/ratings_with_text_ba.txt')

PICKLE_DATA_PATH = './src/data'
