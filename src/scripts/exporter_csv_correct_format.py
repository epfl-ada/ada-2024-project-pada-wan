import numpy as np
def custom_quote(x): #fction that puts all strings inside a ""
    return '"' + x + '"' if isinstance(x, str) else x

def export_csv_correct_format(dataset, filename):
    #put all strings from the datasets beer_consumption_country_wikipedia and country_analysis3 inside a ""
    data_copy=dataset.copy()
    data_copy=data_copy.applymap(custom_quote)
    data_copy.replace('', np.nan, inplace=True)
    data_copy.to_csv(f"data/{filename}.csv", index=False, na_rep='NaN')
