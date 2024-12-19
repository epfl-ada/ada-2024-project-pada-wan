import numpy as np
import pandas as pd
import csv

def export_csv(dataset: pd.DataFrame, filename: str) -> None:
    copy_data = dataset.copy()
    copy_data.replace('', np.nan, inplace=True)
    copy_data.to_csv(f"{filename}.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep='NaN')


if __name__=="__main__":
    df = pd.read_csv('country_analysis3.csv')
    export_csv(df, 'country_analysis3_test')
