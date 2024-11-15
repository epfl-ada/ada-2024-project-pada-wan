import pandas as pd

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