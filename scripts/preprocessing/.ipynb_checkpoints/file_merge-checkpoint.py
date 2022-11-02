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
    df = df.dropna()
    # Marca de tiempo a día, mes, año, hora y minuto
    df = df.drop_duplicates()
    df[["day", "month", "year"]] = df['Timestamp'].str.split('/',expand=True)
    df[["year", 'minute']] = df['year'].str.split(' ',expand=True)
    df[["hour", 'minute']] = df['minute'].str.split(':',expand=True)
    # Datos de string a número
    df = df.replace(',','.', regex=True)
    df[["POZ_PIT_1501A", "POZ_PIT_1401B", "POZ_PIT_1400A", "POZ_PIT_1400B", "day", "month", "year", "minute", "hour"]] = df[["POZ_PIT_1501A", "POZ_PIT_1401B", "POZ_PIT_1400A", "POZ_PIT_1400B", "day", "month", "year", "minute", "hour"]].apply(pd.to_numeric)
    return df


min_data = merge_data(files)
min_data.to_csv("../../data/merged/min_data.csv", index=False)
min_data

for i, month in enumerate(min_data["month"].unique()):
    min_data[min_data["month"] == month].to_csv("../../data/merged/"+ files[i] + ".csv", index=False)

# ## Xlsx to csv daily data

daily_data = pd.read_excel("../../data/original/Consumo en BTU de las bombas con su respectiva estampa.xlsx")
daily_data.to_csv("../../data/merged/daily_data.csv", index=False)
daily_data
