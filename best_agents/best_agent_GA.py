individual=[14.047248847782612, 36.03701223619282, 81.16112225688994, 2.411167975515127, 181.95364535786211, 177.87952017970383, 148.48665790632367, 135.44287122786045, 180.73148359544575, 122.62238813564181, 93.20555096492171, 136.61051262170076, 96.62576206028461, 89.44531115703285, 31.928030028939247, 133.01043985411525, 156.00588908419013, 120.69730092771351, 187.36411309801042, 110.2480465080589, 172.41408661939204, 178.14905983395875, 40.856259036809206, 119.69436821527779, 18.26271368190646, 168.70526215061545, 157.6284729409963, 181.24813614413142, 183.10389681719244, 56.48111905902624, 100.54741124622524, 177.56735272705555, 183.05299784988165, 177.77381567284465, 15.772969834506512, 17.95469168573618]
from scipy import interpolate
import sys
import numpy
from math import *
from operator import *
def if_then_else(input, output1, output2):
	if input: return output1
	else: return output2
distance = 0
prev = 0
t = 0
import random
checkpoints = int (input())
targets = []
def getAngle(a, b, c):
	ang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
	return ang + 360 if ang < 0 else ang
def protectedDiv(left, right):
	try: return left / right
	except ZeroDivisionError: return 1
def fitness(x, y, index):
	global t
	total = dist((x,y), targets[index])
	for i in range(index+1, checkpoints-1):
		total += dist(targets[i], targets[i+1])
	return total + t
def bound(low, high, value):
	return max(low, min(high, value))
for i in range (checkpoints) :
	targets += [[int (j) for j in input(). split()]]
targets = targets * 2
while True:
	checkpoint_index, x, y, vx, vy, angle = [int(i) for i in input().split()]
	print('['+str(fitness(x,y,checkpoint_index))+','+str(checkpoint_index)+','+str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(angle)+']', file = sys.stderr )
	ctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) 
	x_=ctr[:,0] 
	y_=ctr[:,1] 
	tck,u = interpolate.splprep([x_,y_],k=2,s=0)
	u=numpy.linspace(0,1,num=50,endpoint=True)
	out = interpolate.splev(u,tck)
	X = out[0][1]
	Y = out[1][1]
	friction = 0.15
	max_angle = radians(18)
	max_thrust = 200
	thrust=individual[int((angle-1)/10)]
	t+=1
	
	print(str(int(X)) + ' '+ str(int(Y)) + ' ' +str(int(thrust))+ ' agent')