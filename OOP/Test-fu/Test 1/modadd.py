def add(x):
   
    while x < 60: # wird solange ausgefuehrt, wie die Bedingung x <= 4 war ist
        print(x)   # wenn Bedingung erfuellt, dann springt hier her.
        x = x + 2 # wenn hier statt + ein - eigefuegt wird, kommt es zur Endlosschleife

def hello(u):
    u = u + u  # hier kann jede Operation ausgefuehrt werden 
    return u