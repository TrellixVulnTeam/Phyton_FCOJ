for i in range(0,20):
    if i == 15:
        break
    print(i)  # Es wird nur noch 14 ausgegeben, print(i) nicht mehr ausgefuehrt
print('finsch')

for i in range(0,20):
    if i > 15:
        break
    print(i)  # es wird bis 15 ausgegeben
print('finsch')

for i in range(0,20):
    if i < 15:
        break
    print(i)  # Es wird sofort rasgesrungen, print(i) nie ausgefuehrt
print('finsch')