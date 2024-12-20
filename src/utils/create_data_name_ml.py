import pandas as pd

#BeerAdvocate
beers_df_BA = pd.read_csv('./BeerAdvocate/beers.csv')
breweries_df_BA = pd.read_csv('./BeerAdvocate/breweries.csv')

# Merge to have brewery location
merged_df_BA = pd.merge(beers_df_BA, breweries_df_BA, left_on='brewery_id', right_on='id', how='inner')
# keep desired columns
result_df_BA = merged_df_BA[['beer_name', 'brewery_id', 'avg', 'nbr_matched_valid_ratings', 'location']]

#RateBeer
beers_df_RB = pd.read_csv('./RateBeer/beers.csv')
breweries_df_RB = pd.read_csv('./RateBeer/breweries.csv')
merged_df_RB = pd.merge(beers_df_RB, breweries_df_RB, left_on='brewery_id', right_on='id', how='inner')
result_df_RB = merged_df_RB[['beer_name', 'brewery_id', 'avg', 'nbr_matched_valid_ratings', 'location']]

# Merge the two dataframes
result_df = pd.concat([result_df_BA, result_df_RB])

# Save or display the resulting dataframe
result_df.to_csv('merged_beer_breweries.csv', index=False)

