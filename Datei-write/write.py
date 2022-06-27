if __name__ == "__main__":
    f = open('text', 'r+') # w,a,r+,r,b
    f.write ("\n dritte Zeile")
    f.close()
#schreibt aber Text fehlt
# schreibt nicht, aber mit w ueberschreibt aber leer
# nach f.close() muss neu geoeffnet werden