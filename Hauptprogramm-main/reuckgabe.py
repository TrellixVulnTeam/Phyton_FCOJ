variablenWert = "außerhalb der Funktion"
print("Variablenwert vor Funktion:", variablenWert)
def bspfunktion():
    print("Variablenwert in Funktion 1:", variablenWert)
    variablenWert = "IN der Funktion"
    print("Variablenwert in Funktion 2:", variablenWert)

bspfunktion()
print("Variablenwert nach Funktion:", variablenWert)