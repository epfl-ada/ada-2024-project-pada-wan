#convert txt to csv by keeping only the columns we need
# Define file paths and column names
input_file_path = 'src/data/data_BeerAdvocate/ratings.txt'
output_file_path = 'src/data/data_BeerAdvocate/ratings.csv'
columns = [
    'beer_name', 'beer_id', 'brewery_name', 'brewery_id', 'style', 'abv', 'date',
    'user_name', 'user_id', 'appearance', 'aroma', 'palate', 'taste', 
    'overall', 'rating'
]  # we exclude "text" and "review" 

def parse_entry(lines):
    entry = {}
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            if key not in ['text', 'review']: 
                entry[key] = value.strip()
    return entry

data = []
entry_lines = []

with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():  
            entry_lines.append(line.strip())
        else:
            if entry_lines:  
                data.append(parse_entry(entry_lines))
                entry_lines = []

    if entry_lines:
        data.append(parse_entry(entry_lines))


df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"Data saved to {output_file_path}")