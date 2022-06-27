class Fahrzeug():    
    """ Klasse für das Erstellen von Personenkraftwagen """    
    def __init__(self, farbe, baujahr, kmstand, sitze, marke, benzin):        
        """ Eigenschaften farbe, baujahr, kmstand, Sitzplätze, Marke erfassen """        
        self.farbe   = farbe        
        self.baujahr = baujahr        
        self.kmstand = kmstand        
        self.sitze   = sitze        
        self.marke   = marke
        self.benzin  = benzin    
    def hupen(self):        
        """ hier sollte noch eine MP3-Datei ausgegeben werden """        
        print("Trööt")    
    def fahren(self, km):        
        """ wie viele KM gefahren werden, was dem Tachostand aufaddiert wird """        
        self.kmstand += km          
        print("Ich fahre ", km, " Kilometer mit dem ", self.marke)        
        print("Insgesamt bin ich ", self.kmstand ," km mit diesem ", self.marke," gefahren")    
    def parken(self):        
        """ neben fahren schon das größere Problem in Städten """        
        print("Ich habe eine Parkplatz gefunden")    
    def kilometerstand(self):      
        """ Ausgabe des KM-Standes vom Tacho """        
        print("Ich habe ", str(self.kmstand) ," km auf dem" , self.marke, "Tacho")
        """ str stringelement nicht noetig geht auch ohne """
    def tank (self, km):
        self.benzin = km * 0.07
        print("Es wurden", int(self.benzin), "Liter Benzin Super E5 verbraucht")

trabi = Fahrzeug("gruen", 1981, 143000, 8, "Trabi",100)
""" Neues Objekt der Klasse Fahrzeuge.
Die 100 am ende ist noetig, aber vom Wert egal"""

class LKW (Fahrzeug):
    """ 
    Klasse für das Erstellen von Personenkraftwagen 
    ladeflaeche unter def __init__ hinzu
    und in self.ladeflaeche + ladeflaeche
    jetz hat die Kindclasse eine weitere Eingenschaft
    """    
    def __init__(self, farbe, baujahr, kmstand, sitze, marke ,benzin, ladeflaeche):
        super(). __init__ (farbe, baujahr, kmstand, sitze, marke, benzin)
        self.ladeflaeche = ladeflaeche
    
    """ ladeflaeche wird geprintet"""
    def flaeche (self, ladeflaeche):
        self.ladeflaeche = ladeflaeche
        print("die Ladeflaeche betraegt: ", self.ladeflaeche)

    def parken(self):        
        """ neben fahren schon das größere Problem in Städten """        
        print("Ich habe den ", self.marke, "auf dem Firmenhof geparkt")   

ello = LKW("rot", 1980, 123000, 2, "Ello", 1, 10)
panzer = LKW("rot", 1980, 123000, 2, "T34", 1, 10)
""" Neues Objekt der Klasse LKW.
Die 100 am ende ist noetig, aber vom Wert egal"""

panzer.hupen()
trabi.hupen()
ello.hupen()
trabi.kilometerstand()
trabi.fahren(5000)
trabi.kilometerstand()
trabi.parken()
trabi.tank(5000)
ello.fahren(7000)
ello.flaeche(300)
ello.parken()
panzer.fahren(14000)
panzer.flaeche(600)
panzer.parken()
panzer.fahren(14000)
panzer.kilometerstand()