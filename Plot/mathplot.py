import matplotlib.pyplot as plt

ywerte = [4, 7, 1, 9, 5, 2, 8]
xwerte = [1, 2, 3, 4, 5, 6, 7]
plt.plot(xwerte, ywerte)
plt.scatter(xwerte, ywerte, color='red')
plt.xlabel("X-Werte")
plt.ylabel("Y-Werte")

x1 = 1
x2 = 7
y1 = 3.5
y2 = 4.5
plt.plot((x1, x2), (y1, y2))

plt.show()