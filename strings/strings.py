v = 'one ist ein string'

print (v[1]) # strings sind immer auch listen

#v[0] = 'a' ### v[0] = 'a'
            ###     TypeError: 'str' object does not support item assignment

#print (v[0])  kompletter String ist ueberschreibbar

print (v[-1])
print (v[0:3]) # von 0 mit insgesammt 3 Buchstaben
print (v[0:2])
print (v[-4:]) # [-4:10] geht nicht [-4:-10] auch nicht
print (v[:2])
print (v[:6])

v = v + ' du'
print (v)