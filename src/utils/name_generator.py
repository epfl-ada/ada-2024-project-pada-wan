import pandas as pd
import markovify
import numpy as np
import csv

# Load your dataset
df = pd.read_csv('./merged_beer_breweries.csv')
#rename the columns
df.columns = ['Beer_name', 'Brewery_id', 'Rating', 'Number_of_ratings', 'Origin']
# Keep only the columns we need
df = df[['Beer_name', 'Rating', 'Origin']]

# Clean the data
df['Origin'] = df['Origin'].str.split(',').str[0]
df = df.dropna(subset=['Beer_name', 'Rating', 'Origin'])
df['Origin'] = df['Origin'].str.strip().str.title()  
df['Beer_name'] = df['Beer_name'].str.strip() 
#remove countries with more than 30 characters (to remove weird entries)
df = df[df['Origin'].str.len() <= 30]


grouped = df.groupby('Origin')

def create_markov_model(data, rating_col, name_col, min_samples=100):
    # Filter out invalid names
    data = data[data[name_col].notna() & (data[name_col].str.strip() != "")]
    data[rating_col] = pd.to_numeric(data[rating_col], errors='coerce').fillna(1).astype(int)

    # Weight names by rating
    weighted_names = []
    for _, row in data.iterrows():
        weight = int(row[rating_col]) 
        weighted_names.extend([row[name_col]] * weight)

    if len(weighted_names) < min_samples:
        print(f"Not enough samples for model. Found {len(weighted_names)}, expected at least {min_samples}")
        return None

    # Join weighted names into a single text corpus
    text_corpus = "\n".join(weighted_names)

    # Build Markov model
    try:
        return markovify.NewlineText(text_corpus)
    except KeyError:
        print("Error creating Markov model. Insufficient diversity in text corpus.")
        return None


# Create a Markov model per origin
models = {}
for origin, group in grouped:
    if len(group) > 500:
        print(f"Creating model for origin: {origin}, with {len(group)} samples")
        model = create_markov_model(group, 'Rating', 'Beer_name')
        if model:
            models[origin] = model
        else:
            print(f"No valid model for origin: {origin}")


def generate_beer_names(models, origin, num_names=5):
    if origin not in models:
        print(f"No model available for origin: {origin}")
        return []
    
    model = models[origin]
    return [model.make_sentence(tries=100, max_words=4, max_overlap_ratio=0.65) for _ in range(num_names)]

def export_csv(dataset: pd.DataFrame, filename: str) -> None:
    copy_data = dataset.copy()
    copy_data.replace('', np.nan, inplace=True)
    copy_data.to_csv(f"{filename}.csv", index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC, na_rep='NaN')

# Generate names for each origin
generated_names = {}
for origin in models:
    new_names = generate_beer_names(models, origin, num_names=200)
    generated_names[origin] = new_names

# Save the generated names to a CSV file
output_df = pd.DataFrame(generated_names)

export_csv(output_df, 'generated_beer_names')



