def fib (n = 7): # (i,j,n=5) die default i und j immer links angeben
    if n<2:
        return n
    else:
        return fib(n-1)+fib(n-2)


var = fib() # Aufruf ohne Argument , mit (2,3) uebergeben.
#funktioniert hier nicht weil i und j  nicht definiert
# wenn in fib(12) dann wird fuer 12 ausgegeben werden
print(var)

# L.append(5)   listenfunktion 5 Elemente der Liste L = [a,b,c]
# return         Listen mussen gleiche Typen enthalten