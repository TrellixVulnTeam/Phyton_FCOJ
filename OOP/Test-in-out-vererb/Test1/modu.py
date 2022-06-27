def add(x):
	x = x*x
	return x
if __name__ == "__main__": # alles muss danach eingerueckt werden
    # Nur beim direkten aufruf wird der Code unten ausgefuehrt
    # beim import in In-out-r-w.py wird wegen if __name__ == ..... uebergangen

    
    print("hello")  # wird nur ausgegeben wenn modu als ausfuehrbare Datei aufgerufen wird
    x = add(5)
    print(x)
