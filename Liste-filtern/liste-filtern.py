# Liste filtern nach geraden Zahlen

liste1 = [1, 2, 4, 5, 7, 8, 10, 22, 44, 33]
liste2 = list(filter(lambda x: x % 2 == 0, liste1))
liste3 = list(filter(lambda x: x % 2 == 1, liste1))
print(liste2)  # gerade Zahlen werden uebergeben
print(liste3)  # ungerade Zahlen werden uebergeben
