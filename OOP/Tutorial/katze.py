class Tier():
    """ Klasse für das Erstellen von Säugetieren """

    def __init__(self, rufname, farbe, alter):
        self.rufname = rufname
        self.farbe   = farbe
        self.alter   = alter
        self.schlafdauer = 0

    def tut_schlafen(self, dauer):
        print(self.rufname, " schläft jetzt ", dauer , " Minuten ")
        self.schlafdauer += dauer
        print(self.rufname, " Schlafdauer insgesamt: ", self.schlafdauer, " Minuten ")

    def tut_reden(self, anzahl = 1):
        print(self.rufname, "sagt: ", anzahl * "miau ")

class BauplanKatzenKlasse(Tier):
    """ Klasse für das Erstellen von Katzen """

    def __init__(self, rufname, farbe, alter):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(rufname, farbe, alter)

class Hund(Tier):
    """ Klasse für das Erstellen von Hunden """

    def __init__(self, rufname, farbe, alter):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(rufname, farbe, alter)
        
    def tut_reden(self, anzahl = 1):
        print(self.rufname, "sagt: ", anzahl * "WAU ")

katze_sammy = BauplanKatzenKlasse("Sammy", "orange", 3)
hund_bello = Hund("Bello", "braun", 5)

katze_sammy.tut_reden(12)
hund_bello.tut_reden(16)
hund_bello.tut_schlafen(15)