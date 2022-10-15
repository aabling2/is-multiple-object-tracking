import os
import json


# Carrega arquivo de dados JSON
def load_json_config(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name) as json_file:
                data = json.load(json_file)
            return data

    except Exception as ex:
        print(ex)

    print('Configuration file not loaded "options.json!"')
    return None
