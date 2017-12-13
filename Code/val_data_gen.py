import os
from shutil import copyfile
import random as rand

def shuffle(A):
	n = len(A)
	for i in range(n):
		j = rand.randint(i,n-1)
		A[i],A[j] = A[j],A[i]

A1 = os.listdir("Type_1/")
A2 = os.listdir("Type_2/")
A3 = os.listdir("Type_3/")

shuffle(A1)
shuffle(A2)
shuffle(A3)

a1 = len(A1)
a2 = len(A2)
a3 = len(A3)

b1 = int(0.3*a1)
b2 = int(0.3*a2)
b3 = int(0.3*a3)

A1 = A1[:b1]
A2 = A2[:b2]
A3 = A3[:b3]

if not os.path.exists("data/"):
	os.makedirs("data/")

file1 = open("data.txt","w")
to_path = "data/"

for i in A1:
	file1.write(i + " 1\n")
	copyfile("Type_1/"+i,to_path+i)
	os.remove("Type_1/"+i)


for i in A2:
	file1.write(i + " 2\n")
	copyfile("Type_2/"+i,to_path+i)
	os.remove("Type_2/"+i)

for i in A3:
	file1.write(i + " 3\n")
	copyfile("Type_3/"+i,to_path+i)
	os.remove("Type_3/"+i)

file1.close()
