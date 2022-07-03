import pandas as pd
import numpy as np
#pd.set_option("display.max_columns", None)
#import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Dateiname soll angepasst werden
# Parameter zum Auslesen der CSV sollen bei Bedarf angepasst werden (s. zwei Bespielparameter)
# KM: Stundenwerte des Stromverbrauchs der Kältemaschinen (kWh)
# T : Außenlufttemperatur (°C) 
# F : Absolute Außenluftfeuchte (g/kg)
# E : spezische Enthalpie der Außenluft (kJ/kg)
# G : Globalstrahlung (W/m²)

file = "energydata_complete.csv"

# Auslesen der CSV-Datei + Erstellung Dataframe (df)

#df = pd.read_csv(file, parse_dates=['date'], decimal=',')
df = pd.read_csv(file, parse_dates=['date'], decimal=',', sep=';')

# Kleinbuchstaben für Spaltennamen (optional)
df.columns = [x.lower() for x in df.columns]

# Setze Spalte 'date' als "Datetime-Index" für Dataframe
df = df.set_index('date')

# Ausdruck Datensatz
print(df)

# Statistische Kennwerte des Datensatz
df.describe()
#count= Anzahl der Werte
#mean= Durchschnitt der Werte
#std= Standardabweichung der Werte
#min= minimaler Wert
#25%= 25%-Quantil (25 % aller Werte liegen unterhalb dieses Wertes)
#50%= 50%-Quantil (50 % aller Werte liegen unterhalb dieses Wertes)
#75%= 75%-Quantil (75 % aller Werte liegen unterhalb dieses Wertes)
#max= maximaler Wert


# Visualisierung es Ausgabewert

#Anpassung der Diagramgröße (optional)

#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
#plt.figure(figsize=(10,6))

plt.plot(df.km)

plt.xticks( rotation='45')
plt.xlabel('Datum')
plt.ylabel('Stromverbrauch Kältemaschinen [kWh]')