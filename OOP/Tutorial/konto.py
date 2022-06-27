class Konto:
    """ unsere kleines Bankprogramm zum Verwalten Konten/Geld """
    __geldbestand = 0

    def __init__(self, kontonummer, kontostand=0):
        self.__kontonummer = kontonummer
        self.__kontostand  = kontostand

    def geld_abheben(self, betrag):
        print("Geld wird abgehoben:", betrag, " Euro", self.__kontonummer, " Kontonummer")
        self.__kontostand -= betrag
        Konto.__geldbestand -= betrag

    def geld_einzahlen(self, betrag):
        print("Geld wird eingezahlt:", betrag, " Euro",self.__kontonummer, " Kontonummer")
        self.__kontostand += betrag
        Konto.__geldbestand += betrag

    def kontostand_anzeigen(self):
        print("aktueller Kontostand: ", self.__kontostand," Euro"," Kontonummer",self.__kontonummer)
        print("aktueller Geldbestand der Bank: ", Konto.__geldbestand," Euro", "\n")

    def kontostand_aktuell(self):
        return self.__kontostand


class Pluskonto(Konto):
    """ ein Konto, dass nicht überzogen werden kann """

    def __init__(self, kontonummer, kontostand=0):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(kontonummer, kontostand=0)
        """ Auf kontonummer hier kein Zugriff"""
    def geld_abheben(self, betrag):
        print("Geld soll vom Pluskonto abgehoben werden:", betrag, " Euro")
        print("Maximal verfügbar ist gerade:", self.kontostand_aktuell(), " Euro")

        if self.kontostand_aktuell() - betrag >= 0:
            print("Auszahlen von Pluskonto: ", betrag," Euro")
            super().geld_abheben(betrag)
        else:
            print("Sorry, Konto kann nicht überzogen werden!")

kunde_adam = Pluskonto("01")
kunde_eva  = Pluskonto("02")


kunde_adam.kontostand_anzeigen()
kunde_adam.geld_einzahlen(200)
kunde_adam.geld_abheben(101)
kunde_adam.kontostand_anzeigen()
kunde_eva.kontostand_anzeigen()
kunde_eva.geld_einzahlen(2000)
kunde_eva.geld_abheben(401)
kunde_eva.kontostand_anzeigen()
kunde_adam.kontostand_anzeigen()
kunde_adam.geld_einzahlen(20000)
kunde_adam.geld_abheben(1010)
kunde_adam.kontostand_anzeigen()
kunde_eva.kontostand_anzeigen()
kunde_eva.geld_einzahlen(20000)
kunde_eva.geld_abheben(500)
kunde_eva.kontostand_anzeigen()