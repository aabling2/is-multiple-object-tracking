import os
import json


# Cria pasta pelo nome fornecido
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name, exist_ok=True)
            print('-->Pasta "' + folder_name + '" criada com sucesso!')
        except Exception:
            print("Não foi possível criar a pasta:", folder_name)


# Salva arquivo de dados JSON
def save_json(file_name, data, op='w', indent=True, message=False):
    file_name += '.json'
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    try:
        with open(file_name, op) as json_file:
            if indent:
                json.dump(data, json_file, indent=4)
            else:
                json.dump(data, json_file)
        if op != 'a':
            if message:
                print('Dados salvos em: ' + file_name)
    except Exception as ex:
        print(ex)


# Carrega arquivo de dados JSON
def load_json(file_name, message=False):
    file_name += '.json'
    try:
        if os.path.exists(file_name):
            with open(file_name) as json_file:
                data = json.load(json_file)
                if message:
                    print('Dados carregados de: ' + file_name)
            return data
        else:
            print(f'Caminho "{file_name}" não existe!')
            return None

    except Exception as ex:
        print(ex)
        return None
