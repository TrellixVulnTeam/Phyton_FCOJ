# %% [markdown]
# # 1. Datenimport, -darstellung und -visualisierung

# %% [markdown]
# ## 1.1 Import der Funktionsbibliotheken

# %%
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



# %% [markdown]
# ## 1.2 Import der CSV-Datei

# %%
# Dateiname soll angepasst werden
# Parameter zum Auslesen der CSV sollen bei Bedarf angepasst werden (s. zwei Bespielparameter)

# %%
# Date;z2_AC1(kW);z2_Light(kW);z2_Plug(kW)
# KM: z2_AC1(kW)
# T : z2_Light(kW)
# F : z2_Plug(kW)
# E : spezische Enthalpie der Außenluft (kJ/kg)
# G : Globalstrahlung (W/m²)

# %%
file = "data1.csv"

# Auslesen der CSV-Datei + Erstellung Dataframe (df)

#df = pd.read_csv(file, parse_dates=['date'], decimal=',')
df = pd.read_csv(file, parse_dates=['date'], decimal='.', sep=';')

# Kleinbuchstaben für Spaltennamen (optional)
df.columns = [x.lower() for x in df.columns]

# Setze Spalte 'date' als "Datetime-Index" für Dataframe
df = df.set_index('date')

# %% [markdown]
# 

# %% [markdown]
# ## 1.3 Ausdruck vom Datensatz

# %%
print(df)

# %% [markdown]
# ## 1.4  Statistische Kennwerte des Datensatzes

# %%
df.describe()
#count= Anzahl der Werte
#mean= Durchschnitt der Werte
#std= Standardabweichung der Werte
#min= minimaler Wert
#25%= 25%-Quantil (25 % aller Werte liegen unterhalb dieses Wertes)
#50%= 50%-Quantil (50 % aller Werte liegen unterhalb dieses Wertes)
#75%= 75%-Quantil (75 % aller Werte liegen unterhalb dieses Wertes)
#max= maximaler Wert

# %% [markdown]
# ## 1.5 Visualisierung des Ausgabewerts

# %%
#Anpassung der Diagramgröße (optional)

#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
#plt.figure(figsize=(10,6))

#plt.plot(df.km)

#plt.xticks( rotation='45')
#plt.xlabel('Datum')
#plt.ylabel('Stromverbrauch Kältemaschinen [kWh]')

# %% [markdown]
# # 2. Datenvorverarbeitung

# %% [markdown]
# ## 2.1 Ausreißererkennung und -entfernung

# %% [markdown]
# ### 2.1.1 Detektion von Ausreißern

# %%
# Sortiere Ausgabewert nach Größe (absteigend)
sorted_km_up = df.sort_values('km',ascending=False)
upper_treshold=sorted_km_up.km[len(sorted_km_up.head(len(sorted_km_up)//1000))]
# Sortiere Ausgabewert nach Größe (aufsteigend)
sorted_km_down = df.sort_values('km',ascending=True)
lower_treshold=sorted_km_down.km[len(sorted_km_down.head(len(sorted_km_down)//1000))]

# %%
# Boxplot
sns.set(style="whitegrid")
ax = sns.boxplot(df.km)

# %%
print("Das 99,9%-Perzentil des Ausgabewerts beträgt",
      upper_treshold,"kWh,",len(sorted_km_up.head(len(sorted_km_up)//1000)),"Werte liegen oberhalb diesen Wertes.")

# %%
print("Das 0,01%-Perzentil des Ausgabewerts beträgt",
      lower_treshold,"kWh,",len(sorted_km_down.head(len(sorted_km_down)//1000)),"Werte liegen unterhalb diesen Wertes.")

# %% [markdown]
# ### 2.1.2 Entfernung der Ausreißer

# %%
# Entferne Werte aus dem Datensatz ober und unterhalb der festgelegten Grenze
df = df.dropna()
df = df.drop(df[(df.km>upper_treshold)|(df.km<lower_treshold)].index)

# %% [markdown]
# ### 2.1.3 Visualisierung des bereinigten Ausgabewerts

# %%
#Anpassung der Diagramgröße (optional)
#plt.figure(figsize=(16,6))

#plt.plot(df.km)
#plt.xticks( rotation='45')
#plt.xlabel('Datum')
#plt.ylabel('Stromverbrauch Kältemaschinen [kWh]')

# %% [markdown]
# ## 2.2 Merkmalsextraktion

# %%
# Hänge weitere Spalten an den Dataframe via Datetime-Index

df['month'] = df.index.month
df['week'] = df.index.week

df['weekday'] = df.index.weekday
df['hour'] = df.index.hour

print (df)

# %% [markdown]
# ## 2.3 Datenaggregation und -visualisierung

# %%
# Durchschnittswert je Tagesstunde
def hourly(x,df=df):
    return df.groupby('hour')[x].mean()

# Durchschnittswert je Wochentag
def daily(x,df=df):
    return df.groupby('weekday')[x].mean()
    
# Durchschnittswert je Wochentag und Monat
#def monthly_daily(x,df=df):
#    by_day = df.pivot_table(index='weekday', 
#                                columns=['month'],
#                                values=x,
#                                aggfunc='mean')
#    return round(by_day, ndigits=1)
#
# Durchschnittswert je Wochentag und Monat
def daily_hourly(x,df=df):
    by_day_hour = df.pivot_table(index='weekday', 
                                columns=['hour'],
                                values=x,
                                aggfunc='mean')
    return round(by_day_hour, ndigits=0)


# Durchschnittswert je Wochentag und Tagesstunde
def code_mean(data, cat_feature, real_feature):
    return dict(data.groupby(cat_feature)[real_feature].mean())

df['weekday_avg'] = list(map(code_mean(df[:], 'weekday', "km").get, df.weekday))
df['hour_avg'] = list(map(code_mean(df[:], 'hour', "km").get, df.hour))
df.describe()

# %% [markdown]
# ### 2.3.1 Mittelwert je Wochentag

# %%
# Diagramm: Mittlerer Stromverbrauch Kältemaschinen je Wochentag

ticks = list(range(0, 7, 1)) 

daily('km').plot(kind = 'bar', figsize=(10,8))

labels = "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split()
plt.xlabel('Wochentag')
plt.ylabel('Stromverbrauch Kältemaschinen [kWh]')
plt.title('Mittlerer Stromverbrauch Kältemaschinen je Wochentag')
plt.xticks(ticks, labels,rotation='45')

# %% [markdown]
# ### 2.3.3 Mittelwert je Wochentag und Tagesstunde

# %%
# Diagramm: Mittlerer Stromverbrauch Kältemaschinen je Tagesstunde

hourly('km').plot(figsize=(10,8))
plt.xlabel('Tagesstunde')
plt.ylabel('Stromverbrauch Kältemaschinen [kWh]')
ticks = list(range(0, 24, 1))
plt.title('Mittlerer Stromverbrauch Kältemaschinen je Tagesstunde')

plt.xticks(ticks)

# %% [markdown]
# ### 2.3.4 Mittelwert je Wochentag und Stunde (Heatmap)

# %%
# Heatmap: Mittlerer Stromverbrauch Kältemaschinen je Wochentag und Stunde
sns.set(rc={'figure.figsize':(15,12)},)
ax=sns.heatmap(daily_hourly('km').T,cmap="YlGnBu",
               xticklabels="Mo Di Mi Do Fr Sa So".split(),
               yticklabels=list(range(0, 24, 1)),
               annot=True, fmt='g',
               cbar_kws={'label': 'Stromverbrauch Kältemaschinen [kWh]'}).set_title("Mittlerer Stromverbrauch Kältemaschinen je Wochentag und Stunde").set_fontsize('15')
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.show()



