import threading
import modu
class MyClass:
    a = 15          # DIESE HABEN UMFASSENDEN gELTUNGSBEREICH IN DER INSTANZ/KLASSE
    string = "ich"
    z = "Du"
    list = [] # keine einfache Variable und ist fuer alle Instanzen gueltidg
              # man kann unendlich viele Variablen definieren
    def do_some (self, neu_a):  # ohne self ist es eine lokale Variable
        self.a = neu_a          # self macht in Classe verfuegbar
    def __init__(self,neuzahl): # self immer innerhalb der Class noetig egal Funk oder Konstruktor
        self.wort = " ich bin aus Kontructor "
        list = []      # ist nur fuer die aufgerufene Instanz hier gueltig
#   (no funz)         def do_next (self, alt_a): # ohne self ist es eine lokale Variable
#                     self.neu_a = alt_a         # self macht in Classe verfuegbar
        
instanz = MyClass(121243) # ist neuzahl
instanz.do_some(1234)     #  ist neu_a kann strin oder int "ich" eingeben
instanz2 = MyClass(55)    # ruft die Klasse mit anderem Parameter auf
print(instanz.string)
print(instanz.z)
print(instanz2.a)
print(instanz.wort)
print(type(instanz.z))
print(type(instanz.a))
print(type(instanz.wort))
y = input("gib x ein : ")
x = int(y)   # von str nach int
# x = str(a..) # von int nach str   // FUNKT NATUERLICH
x = modu.add(x)
print(x)
print(instanz2.a)