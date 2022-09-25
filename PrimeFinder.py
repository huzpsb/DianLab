myList = [2, 3]
myInt = 3
while True:
    myInt += 2
    if myInt > 1000:
        break
    for toCheck in myList:
        if myInt % toCheck == 0:
            continue
    myList.append(myInt)
for toPrint in myList:
    print(toPrint)
