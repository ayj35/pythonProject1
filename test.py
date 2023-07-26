import pandas as pd
import os

path_dir = 'load_raw'
file_list = os.listdir(path_dir)
dataframes = []

for file in file_list:
    print(file)
    file_path = os.path.join(path_dir,file)
    print(file_path)
    df = pd.read_excel(file_path)
    dataframes.append(df)
    break
print(dataframes[0])