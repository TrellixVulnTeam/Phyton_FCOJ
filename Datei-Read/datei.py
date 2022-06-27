if __name__ == "__main__":
    f = open('text.txt', 'w') # w,a,r+,r,b
    print(f.read())
    print('hello')
    print(f.readline()) # diese 1.Zeile wird nur wenn in 3 ausgeblendet
    
    print(f.read(2))
    
#    for line in f:  #die beiden oben funken auch, aber nur jewels einzeln,???
#        print(line) # 2 Leerzeilen wegen eine in Datei und eine durch (line)