import datetime
import time

print(datetime.datetime.now().strftime("%H:%M:%S-%d.%m.%Y")) #B,b.W,w moeglich

print(datetime.date.today().strftime("%H:%M:%S-%d.%m.%Y")) # gibt dann keine Zeit zurueck   B,b.W,w moeglich

print(time.time()) #Zeit seit 1950 ?? die vergangen ist

# time kann Differenzen berechnen
start = time.time()
time.sleep(10)         # wartet 10 sek
end = time.time()
print(end-start)
