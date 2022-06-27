'''
Multithreading
- Shared Memory
- UIs
- I/O
- Hyper Threading
- Threads spawnen schneller
- Python ist zuständig

Multiprocessing
- Eigener Memory Space
- eigener CPU Kern
- Kindprozesse kann man töten
- Prozesse brauchen länger bis sie erstellt sind
- System verwaltet
'''
import multiprocessing
import math
import threading
import random
import time
import sys


"""
"res"
Yes, "res" is a common shortened version of "result".
Sometimes folks will shorten a variable name too much
so that it actually loses some readability.
"""
def factorize(num):
    res = []
    i = 2
    while True:
        if num == 1:
            return res
        rest = num % i
        if rest == 0:
            res.append(i)
            num = num // i
        else:
            i = i+1
"""
append() bedeuted 
The append() method in python adds a single item to the existing list.
It doesn't return a new list of items but will modify the original list
by adding the item to the end of the list. After executing the method
append on the list the size of the list increases by one.
Syntax. list_name.append(item)
"""
def mt_worker(outdict, numbers):
        for num in numbers:
            outdict[num] = factorize(num)
"""
Chunking bedeuted
Advertisements. Chunking is the process of grouping similar
words together based on the nature of the word. In the below
example we define a grammar by which the chunk must be generated.
The grammar suggests the sequence of the phrases like nouns and adjectives etc.
"""
def mt_factor(numbers, num_threads):
    chunk = int(math.ceil(len(numbers) / num_threads))
    threads = []
    outs = [{} for i in range(num_threads)]
    for i in range(num_threads):
        t = threading.Thread(
            target=mt_worker,
            args=(outs[i], numbers[chunk * i:chunk * (i + 1)]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return {key: v for out_dict in outs for key, v in out_dict.items()}

#{42: [2*7*3], 1337:...}
def mp_worker(queue, numbers):
    result = {}
    for i in numbers:
        result[i] = factorize(i)
    queue.put(result)

def mp_factor(numbers, processes):
    queue = multiprocessing.Queue()
    chunks = int(math.ceil(len(numbers)) / processes)
    procs = []
    for i in range(processes):
        proc = multiprocessing.Process(target=mp_worker, args=(queue, numbers[chunks*i:chunks*(i+1)]))
        procs.append(proc)
        proc.start()

    results = {}
    for i in range(processes):
        results.update(queue.get())

    for i in procs:
        i.join()
    return results
"""
random()
Almost all module functions depend on the basic function random() ,
which generates a random float uniformly in the semi-open range [0.0, 1.0).
... Python uses the Mersenne Twister as the core generator.
"""

if __name__ == '__main__':
    numbers = []
    for i in range(8*1024*16):
        numbers.append(random.randint(1, 5000))
    for procs in [1,2,4,8]:
        start = time.time()
        res = mp_factor(numbers, procs)
        print("Multiprocessing with " + str(procs) +" Processes took " + str((time.time() - start)) + " seconds.")
    for threads in [1,2,4,8]:
        start = time.time()
        res = mt_factor(numbers, threads)
        print("Multithreading with " + str(threads) +" Threads took " + str((time.time() - start)) + " seconds.")
"""
randint()
Basically, the randint() method in Python returns a random integer
value between the two lower and higher limits (including both limits)
provided as two parameters. It should be noted that this method is only
capable of generating integer-type random value.

Das Programm endet nicht. Es laesst 2 Prozesse offen, die bis zu je 25 % Resourcen verbrauchen.

"""