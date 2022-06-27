import modu
import modi

x = 4

#if __name__ == "__main__": #funkt nicht

print(x)
#print(modu.funk1(x))
x = modu.funk1(x)
print("aus Ruegabe x = modu.funk1(x) ist, ",x)
print("printausgabe 144 * 144 ",x*x)

x = modu.funk2(x)

print("aus funk2 x = i , x mal x ist ",x * x)
print(x)
x = modi.funk1(x)
print(x)

x = modi.funk2(x)
print(x)

x = modi.funk3(x)
print(x)

x = modi.funk4(x)
print(x)