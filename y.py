import pandas as pd
import os

# Lista de arquivos CSV a serem unidos
arquivos_csv = ['dados_parte_1.csv', 'dados_parte_2.csv', 'dados_parte_3.csv']  # Adicione os nomes dos arquivos

# DataFrame vazio para armazenar os dados unidos
df_unido = pd.DataFrame()   

# Loop para ler e concatenar os arquivos CSV
for arquivo in arquivos_csv:
    df_temp = pd.read_csv(arquivo)
    df_unido = pd.concat([df_unido, df_temp], ignore_index=True)

# Salva o DataFrame unido em um novo arquivo CSV
df_unido.to_csv('dados_unidos.csv', index=False)

print('Arquivos CSV unidos com sucesso!')
