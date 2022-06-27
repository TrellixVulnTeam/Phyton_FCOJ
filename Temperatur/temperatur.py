# geg: Temperatur in Grad Celsius
# ges: Temperatur in Grad Kelvin
# K = C + 273.15
import modul # die beiden Funtionen ins modul auslagern geht nicht
            # muss probiert werden
            # habe pri als Funtion in modul definiert

def get_temperatur():
    while True:
        C = input( " Gib Grad Celsius ein: ")
        try:
            
            C = float(C)
            return C
        except ValueError:
            print("war keine Temperaur")
            
def convert(C):
    K = C + 273.15
    return K

if __name__ == "__main__":
    
    C = get_temperatur() #ist nicht definiert warum
    print( "Das sind " + str(convert(C)) + " Kelvin")

print(modul.pri())