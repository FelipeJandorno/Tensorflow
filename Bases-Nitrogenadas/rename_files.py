# DICIONÁRIO DO CÓDIGO:
#
# Função search_dir() - Verifica a existência de uma pasta dentro do projeto e, caso a pasta não exista, retorna False
# Função rename_files() - Renomeia todos os arquivos da pasta 'SPA_TO_CSV' e os envia para 'CSV_TEST'
# Função count_files() - Contabiliza a quantidade de arquivos que existem na pasta 'CSV_TEST'

import os
def search_dir(new_path):
    dir_flag = False
    for filename in os.listdir(os.getcwd()):
        if filename == new_path:
            dir_flag = True
    return dir_flag

def count_files(path):
    cnt = 0
    for filename in os.listdir(path):
        cnt = cnt + 1
    return cnt

def rename_files():
    # Armazena o endereço da pasta para onde os arquivos estão
    old_path = "SPA_TO_CSV/"
    # Armazena o endereço da pasta para onde os arquivos irão
    new_path = "CSV"

    # Verifica a existência do diretório e cria um novo, caso necessário
    if not search_dir(new_path):
        print('new folder')
        os.mkdir(new_path)

    # Verifica se o diretório está vazio, caso contrário deleta todos os arquvos presentes no novo diretório
    if count_files(new_path) == 0:
        # Renomeia todos os arquivos de SPA para CSV
        for filename in os.listdir(old_path):
            first_path = old_path + "/" + filename
            filename = filename.replace(".SPA", ".CSV")
            second_path = new_path + "/" + filename
            os.rename(first_path, second_path)
    else:
        for filename in os.listdir(new_path):
            remove_path = new_path + "/" + filename
            os.remove(remove_path)