for i in range(0,20):
    if i == [15,14,13]: #Das mit der Liste funkt nicht, nur 15 die erstewird beruecksichtigt
        continue
    print(i)  # die 15 wird NICHT ausgegeben, denn wird durch continue uebersprungen
print('finsch')

for i in range(0,20):
    if i < 15: #Das mit der Liste funkt nicht, nur 15 die erstewird beruecksichtigt
        continue
    print(i)  # die 15 wird NICHT ausgegeben, denn wird durch continue uebersprungen
print('finsch')

for i in range(0,20):
    if i > 15: #Das mit der Liste funkt nicht, nur 15 die erstewird beruecksichtigt
        continue
    print(i)  # die 15 wird NICHT ausgegeben, denn wird durch continue uebersprungen
print('finsch')