class myClass:
    zahl = 42
    string = "zeichenkette"
    list = [1,2,list]
    var  = 5
    
#    def __init__(self, varneu = "hallo"):
#        self.var = varneu
#        self.list = [23,33]
# Das ist der Konstruktor, das self muss am Anfang hinter init stehen        
    
    def do_something(self, neuzahl):
        self.zahl = neuzahl
        self.list.append(neuzahl)
# self weil es aus dem Konstruktor kommt        

instanz1 = myClass() # es wird jeweils nichts uebergeben
instanz2 = myClass() # es wird jeweils nichts uebergeben
#instanz1.do_something("list") # hier wird zusaetzlich die Funktion def do_something aufgerufen
#instanz2.do_something(5)
print(instanz1.zahl)
print(instanz2.string)
print(instanz1.var)
print(instanz1.list)
print(instanz2.list)
"""
instanzX = ruft die Klasse auf und uebergibt zB. bei instanz2 ("a") als Zeichen
bleibt myClass() leer wird nicht
"""