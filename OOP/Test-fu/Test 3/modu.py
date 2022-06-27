def funk1(x):
    x = x * x
    print("fu x mal x ist ",x)
    return x

def funk2(x):
    for i in range(3,13,2): # 10 wird nicht ausgegeben
                
        print("zaehlen 5 bis 20 mit range(4,21,2) ",i) # in 2 Schritten
        x = i
        print(" x = i  x ist ", x ) # wird jedesmal durchlaufen. Geht schneller ???
    
    return x
       