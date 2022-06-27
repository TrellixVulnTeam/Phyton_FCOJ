class katze():
    """OOP"""
    
    def __init__(self, rufname, farbe, alter):
        self.rufname = rufname        
        self.farbe   = farbe        
        self.alter   = alter
    
    def tut_miauen(self, anzahl = 1):
        print(anzahl * "miau ")

"""print(help(katze))
   print(dir(katze))
    Help on class katze in module __main__:

class katze(builtins.object)
 |  katze(rufname, farbe, alter)
 |  
 |  OOP
 |  
 |  Methods defined here:
 |  
 |  __init__(self, rufname, farbe, alter)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  tut_miauen(self, anzahl=1)
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
(END)
"""
    
katze_sammy = katze("Sammy", "orange", 3)
katze_sammy.tut_miauen(5)
print(katze_sammy.alter)
print(katze_sammy.farbe)
print(katze_sammy.rufname)

class pkw():

    def __init__(self,farbe,baujahr,kmstand,platz = 4,marke = "VW"):
        self.farbe = farbe
        self.baujahr = baujahr
        self.kmstand = kmstand
        self.platz = platz
        self.marke = marke
        
    def tut_fahren(self,km = 1):
        """ wie viele KM gefahren werden, was dem Tachostand aufaddiert wird """        
        self.kmstand += km        
        print("Ich fahre ", km, " Kilometer")        
        print("Insgesamt bin ich ", self.kmstand ," gefahren") 
    
    
    
    def kilometerstand(self):        
        """ Ausgabe des KM-Standes vom Tacho """        
        print("Ich habe ", str(self.kmstand) ," auf dem Tacho")
        
trabant = pkw("rot",1978,12400,8,"Toyota")



print(trabant.baujahr)
print(trabant.farbe)
print(trabant.marke)
print(trabant.platz)

katze_sammy.tut_miauen(10)
trabant.tut_fahren(200)
trabant.kilometerstand()
trabant.tut_fahren(1200)
trabant.tut_fahren(2100)
trabant.kilometerstand()
"""print(dir(trabant))


['__class__', '__delattr__', '__dict__', '__dir__', '__doc__',
 '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__',
  '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
   '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',
    '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
     '__weakref__', 'baujahr', 'farbe', 'km', 'marke', 'platz', 'tut_fahren']"""
