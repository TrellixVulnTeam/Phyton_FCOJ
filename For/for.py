a = 34
aaa =100
x = [3,a,4,5,6,'wort',0,None,aaa,'aaab'] # geht das wirklich, es sollen alle vom gleichen Typ sein. addieren kann man so nicht
# auch Variablen funken siehe a und aaa, 'aaa' aber string
for w in x: # geht die Liste vom ersten bis letzten Element durch
   print(w) # und gibt das defienierte w aus. w ist beliebig 
print('jetzt wurden alle Elemente ausgegeben')
print('Ende')
print(x[9])
print(x[1])
print(x[-1])
print(x[0])
x[0] = -100 # uberschreibt
print(x[0])
# die Liste kann also auch aus unterschiedlichen Elemente bestehen
x = x + [2,3,0] #haengt an
print(x)