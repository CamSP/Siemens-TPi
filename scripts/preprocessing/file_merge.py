import pandas as pd

# # File type preprocessing

# ## Merge month data

# List of files
files = ["AGO2020", 
         "SEP2020", 
         "OCT2020", 
         "NOV2020", 
         "DIC2020", 
         "ENE2021", 
         "FEB2021",
         "MAR2021"]


# File merge function
def merge_data(files):
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv("../../data/original/"+file+".csv", sep=";")
        df = pd.concat([df, data])
    df = df.fillna(method="ffill")
    return df.dropna()


min_data = merge_data(files)
min_data.to_csv("../../data/merged/min_data.csv")
min_data

# ## Xlsx to csv daily data

daily_data = pd.read_excel("../../data/original/Consumo en BTU de las bombas con su respectiva estampa.xlsx")
daily_data.to_csv("../../data/merged/daily_data.csv")
daily_data
