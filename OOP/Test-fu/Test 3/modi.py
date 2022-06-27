def funk1(x):
    if  x < 5:
        x = x + 1
    elif x == 20:
        x = x * x
    else:
        x = x +10
    return x

def funk2(x):
    while x < 50:  # wird solange ausgefuehrt bis die Bedingung false ist.
        x = x + 1
        print("die Whileschleife\n ",x)
    return x

def funk3(x):
    for i in range(5):    # die for schleife ist hauptsaechlich fuer Listen.
       print(" i ist ",i) # range erzeugt eine Liste.// x=[0,1,2,3,4,5] z.B.
    return x
    
def funk4(x):
    x = x + 10
    print(x)
    return x