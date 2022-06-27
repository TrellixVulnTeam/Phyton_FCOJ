class Tier():
    """ Beispiel für Variablen in Klassen """
    tieranzahl = 0

    def __init__(self, rufname):
        self.rufname = rufname

    def anzahl_tiere():
        print("aktuelle Anzahl: ", Tier.tieranzahl)

class Katze(Tier):
    """ Klasse für das Erstellen von Katzen """
    # print("Aus Klasse Katze: ", Tier.tieranzahl)

    def __init__(self, rufname):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(rufname)
        Tier.tieranzahl += 1

class Hund(Tier):
    """ Klasse für das Erstellen von Hunden """
    # print("Aus Klasse Hunde: ", Tier.tieranzahl)

    def __init__(self, rufname):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(rufname)
        Tier.tieranzahl += 1

Tier.anzahl_tiere()
katze_schnurri = Katze("Schnurri")
print(katze_schnurri.rufname)
Tier.anzahl_tiere()

hund_bello = Hund("Bello")
print(hund_bello.rufname)
Tier.anzahl_tiere()

hund_wedler = Hund("Wedler")
print(hund_wedler.rufname)
Tier.anzahl_tiere()