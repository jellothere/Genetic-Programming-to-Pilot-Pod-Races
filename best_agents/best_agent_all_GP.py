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
	if dist((x,y), targets[checkpoint_index]) > 600:
		ctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) 
		x_=ctr[:,0] 
		y_=ctr[:,1] 
		tck,u = interpolate.splprep([x_,y_],k=2,s=0)
		u=numpy.linspace(0,1,num=30,endpoint=True)
		out = interpolate.splev(u,tck)
		X = out[0][1]
		Y = out[1][1]
	else:
		X = vx
		Y = vy
	friction = 0.15
	max_angle = radians(18)
	max_thrust = 200
	angle = radians(getAngle((vx, vy),(x,y),(X,Y)))
	t+=1
	thrust=abs(bound(-200,200,mul(max_angle, add(if_then_else(lt(sub(mul(if_then_else(False, vx, friction), sub(friction, vy)), vy), add(vx, max_thrust)), sin(protectedDiv(vx, sub(protectedDiv(max_angle, vy), cos(max_thrust)))), max_thrust), max_thrust))))
	print(str(int(X)) + ' '+ str(int(Y)) + ' ' +str(int(thrust))+ ' agent')
