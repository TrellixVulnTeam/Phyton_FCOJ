x = 200
print('anfangs ist x = ',x)
if x < 50: # wird ausgefuehrt wenn wahr true
    print("x ist kleiner 50")
    x = 0
    
elif x > 50:
    print("x ist groesser 50")
    x=2
else:
    print(" x ist geleich 50 ")
    x =10
print("es wird die jeweilige Bedingung ausgefuehrt\n und hier geht das Programm weiter\n und x wird ausgegeben\n aber es wurde jeweils ueberschrieben mit Null\n deshalb Ausgabe 0")
print('x = ',x)