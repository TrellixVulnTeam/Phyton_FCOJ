def fib (n):
    if n<2:
        return n
    else:
        return fib(n-1)+fib(n-2)


var = fib(10)
print(var)