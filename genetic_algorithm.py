from py4j.java_gateway import JavaGateway
import requests
import random
from random import randint
from random import choice
import matplotlib.pyplot as plt
import numpy as np
from numpy import*
import multiprocessing
import concurrent.futures
import pickle
import time
import os, glob
from tqdm import tqdm

population = []                                 # Variable global para la población
mutation_rate = 0.006                           # Ratio de mutación por gen
uniform_rate = 0.5                              # Ratio de reproducción
evaluaciones = 0                                # Número de evaluaciones necesarias
calls = dict()                                  # Diccionario de llamadas previas

# Velocidad y ángulo
# program = "from scipy import interpolate\nimport sys\nimport numpy\nfrom math import *\nfrom operator import *\ndef if_then_else(input, output1, output2):\n\tif input: return output1\n\telse: return output2\ndistance = 0\nprev = 0\nt = 0\nimport random\ncheckpoints = int (input())\ntargets = []\ndef getAngle(a, b, c):\n\tang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))\n\treturn ang + 360 if ang < 0 else ang\ndef protectedDiv(left, right):\n\ttry: return left / right\n\texcept ZeroDivisionError: return 1\ndef fitness(x, y, index):\n\tglobal t\n\ttotal = dist((x,y), targets[index])\n\tfor i in range(index+1, checkpoints-1):\n\t\ttotal += dist(targets[i], targets[i+1])\n\treturn total + t\ndef bound(low, high, value):\n\treturn max(low, min(high, value))\nfor i in range (checkpoints) :\n\ttargets += [[int (j) for j in input(). split()]]\ntargets = targets * 2\nwhile True:\n\tcheckpoint_index, x, y, vx, vy, angle = [int(i) for i in input().split()]\n\tprint('['+str(fitness(x,y,checkpoint_index))+','+str(checkpoint_index)+','+str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(angle)+']', file = sys.stderr )\n\tctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) \n\tx_=ctr[:,0] \n\ty_=ctr[:,1] \n\ttck,u = interpolate.splprep([x_,y_],k=2,s=0)\n\tu=numpy.linspace(0,1,num=50,endpoint=True)\n\tout = interpolate.splev(u,tck)\n\tX = out[0][1]\n\tY = out[1][1]\n\tfriction = 0.15\n\tmax_angle = radians(18)\n\tmax_thrust = 200\n\tangle = getAngle((vx, vy),(x,y),(X,Y))\n\tindividual = numpy.array(individual).reshape(6,6,36)\n\tthrust=individual[int(bound(0, 599, vx)/100)][int(bound(0, 599, vy)/100)][int((angle-1)/10)]\n\tt+=1\n\t"
# bounds = [[0, 200.0]] * 36 * 6 * 6 
# Solo velocidad
# program = "from scipy import interpolate\nimport sys\nimport numpy\nfrom math import *\nfrom operator import *\ndef if_then_else(input, output1, output2):\n\tif input: return output1\n\telse: return output2\ndistance = 0\nprev = 0\nt = 0\nimport random\ncheckpoints = int (input())\ntargets = []\ndef getAngle(a, b, c):\n\tang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))\n\treturn ang + 360 if ang < 0 else ang\ndef protectedDiv(left, right):\n\ttry: return left / right\n\texcept ZeroDivisionError: return 1\ndef fitness(x, y, index):\n\tglobal t\n\ttotal = dist((x,y), targets[index])\n\tfor i in range(index+1, checkpoints-1):\n\t\ttotal += dist(targets[i], targets[i+1])\n\treturn total + t\ndef bound(low, high, value):\n\treturn max(low, min(high, value))\nfor i in range (checkpoints) :\n\ttargets += [[int (j) for j in input(). split()]]\ntargets = targets * 2\nwhile True:\n\tcheckpoint_index, x, y, vx, vy, angle = [int(i) for i in input().split()]\n\tprint('['+str(fitness(x,y,checkpoint_index))+','+str(checkpoint_index)+','+str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(angle)+']', file = sys.stderr )\n\tctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) \n\tx_=ctr[:,0] \n\ty_=ctr[:,1] \n\ttck,u = interpolate.splprep([x_,y_],k=2,s=0)\n\tu=numpy.linspace(0,1,num=50,endpoint=True)\n\tout = interpolate.splev(u,tck)\n\tX = out[0][1]\n\tY = out[1][1]\n\tfriction = 0.15\n\tmax_angle = radians(18)\n\tmax_thrust = 200\n\tangle = getAngle((vx, vy),(x,y),(X,Y))\n\tindividual = numpy.array(individual).reshape(60,60)\n\tthrust=individual[int(bound(0, 599, vx)/10)][int(bound(0, 599, vy)/10)]\n\tt+=1\n\t"
# bounds = [[0, 200.0]] * 60 * 60 
# Solo ángulo
program = "from scipy import interpolate\nimport sys\nimport numpy\nfrom math import *\nfrom operator import *\ndef if_then_else(input, output1, output2):\n\tif input: return output1\n\telse: return output2\ndistance = 0\nprev = 0\nt = 0\nimport random\ncheckpoints = int (input())\ntargets = []\ndef getAngle(a, b, c):\n\tang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))\n\treturn ang + 360 if ang < 0 else ang\ndef protectedDiv(left, right):\n\ttry: return left / right\n\texcept ZeroDivisionError: return 1\ndef fitness(x, y, index):\n\tglobal t\n\ttotal = dist((x,y), targets[index])\n\tfor i in range(index+1, checkpoints-1):\n\t\ttotal += dist(targets[i], targets[i+1])\n\treturn total + t\ndef bound(low, high, value):\n\treturn max(low, min(high, value))\nfor i in range (checkpoints) :\n\ttargets += [[int (j) for j in input(). split()]]\ntargets = targets * 2\nwhile True:\n\tcheckpoint_index, x, y, vx, vy, angle = [int(i) for i in input().split()]\n\tprint('['+str(fitness(x,y,checkpoint_index))+','+str(checkpoint_index)+','+str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(angle)+']', file = sys.stderr )\n\tctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) \n\tx_=ctr[:,0] \n\ty_=ctr[:,1] \n\ttck,u = interpolate.splprep([x_,y_],k=2,s=0)\n\tu=numpy.linspace(0,1,num=50,endpoint=True)\n\tout = interpolate.splev(u,tck)\n\tX = out[0][1]\n\tY = out[1][1]\n\tfriction = 0.15\n\tmax_angle = radians(18)\n\tmax_thrust = 200\n\tthrust=individual[int((angle-1)/10)]\n\tt+=1\n\t"
bounds = [[0, 200.0]] * 36 
end_prog = "\n\tprint(str(int(X)) + ' '+ str(int(Y)) + ' ' +str(int(thrust))+ ' agent')"

gateway = JavaGateway()           # connect to the JVM

# Ángulo definido por tres puntos
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

# Rosenbrock objective function
def rosenbrock(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

# Six hump camell back objective function
def six_hump_camel_back(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1 * np.power(x1, 2) + np.power(x1, 4) / 3) * np.power(x1, 2) + x1 * x2 + (
                -4 + 4 * np.power(x2, 2)) * np.power(x2, 2)

# Eggholder objective function
def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    a=sqrt(fabs(x2+x1/2+47))
    b=sqrt(fabs(x1-(x2+47)))
    c=-(x2+47)*sin(a)-x1*sin(b)
    return c

# Eggholder objective function
def cost_function(x):
    # Transform the tree expression in a callable function
    pid = str(os.getpid()) + '.py'
    file='agent'+pid
    with open(file, 'w') as filetowrite:
        filetowrite.write("individual="+str(x)+"\n"+program+end_prog)
    result = gateway.entry_point.simulate(3, pid)  # invoke constructor
    try:
        result = next((dop for dop in reversed(result['0']) if dop is not None), None)
        fitness = eval(result)[0]
    except TypeError:
        print(result)
        fitness = 40000
    
    return fitness
    
# Decodes bitstring to two real numbers (x,y)
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(bounds)):
        # Extract two substrings
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        integer = int(substring, 2)
        # Scaler to desire bounds
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded

# Obtains fitness vallue for an individual
def get_fitness(individual):
    global evaluaciones, calls
    if str(individual) not in calls:
        evaluaciones +=1
        calls[str(individual)] = cost_function(decode(bounds, 32, individual))
    return calls[str(individual)]

#####################################################################################################################################

# Función de obtención del fitness individual
def get_fittest(population):
    minimum = 10000000000000000     # Número arbitariamente grande
    best_solution = population[0]
    for individual in population:
        fitness = get_fitness(individual)
        if fitness < minimum:
            minimum = fitness
            best_solution = individual
    return best_solution
 
# Función para generar aleatoriamente cadenas de bits
def initializePopulation(n):
    global population
    for i in range(0, n):
        population += [''.join(choice('01') for _ in range(32*36))]

# Selección de individuos por torneos
def tournament_selection():
    global population
	#Tournament pool
    tournament = []
    e = int(len(population)*0.05)
    for i in range(0, e):
	    random_id = randint(0, len(population)-1)
	    tournament += [population[random_id]]
    fittest = get_fittest(tournament)
    return fittest

# Operador de mutación
def mutate(individual):
    global mutation_rate
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() <= mutation_rate:
            gene = randint(0,1)
            if gene == 0: 
                individual[i] = '0'
            else: 
                individual[i] = '1'
    return ''.join(individual)

# Operador de cruce uniforme
def crossover(individual1, individual2):
    global uniform_rate
    new_individual = ''
    individual1 = list(individual1)
    individual2 = list(individual2)
    for i in range(len(individual1)):
        if random.random() <= uniform_rate:
            new_individual += individual1[i]
        else:
            new_individual += individual2[i]
    return new_individual

# Se genera una nueva generación aplicando elitismo, selección por torneo, cruce uniforme y mutación	
def evolve_population():
    global population
    # Elitism
    e = int(len(population)*0.06)
    new_population = get_elit()

    for i in range(e, len(population)):
        individual1 = tournament_selection()
        individual2 = tournament_selection()
        new_individual = crossover(individual1, individual2)
        new_population += [new_individual]
   
    for i in range(e, len(population)):
        new_population[i] = mutate(new_population[i])

    return new_population

# Devuelve el fitness medio de un población entera
def getMeanPopulationFitness():
    global population
    total = 0
    for individual in population:
        total += get_fitness(individual)
    return (total/len(population))

# Devuelve la élite/hof
def get_elit():
    global population
    elit = sorted(population, key=lambda individual: get_fitness(individual))
    e = int(len(population)*0.06)
    return elit[:e]

def island(id, solutions, history):
    global population                       # Población de la isla
    m = 20                                 # Tamaño población de la isla
    initializePopulation(m)                 # Inicialización de la población de la isla
    with tqdm(range(10), desc = 'Island ' +str(id),  position=id) as generations:
        for i in generations:
            res = pool.map(get_fitness, population)
            for p, f in zip(population, res): 
                if str(p) not in calls: calls[str(p)]=f
            best = get_fittest(population)
            best_f = get_fitness(best)
            history += [[best_f, getMeanPopulationFitness()]]
            generations.set_postfix(f=best_f)
            population = evolve_population() # Nueva generación
    solutions += get_elit()
    

if __name__ == "__main__":

    start_time = time.time()
    manager = multiprocessing.Manager()  # Manager de los procesos
    solutions = manager.list()           # Creando una lista compartida entre todos los procesos
    history = manager.list()             # Creando una lista compartida entre todos los procesos
    jobs = []                            # Lista con todos los procesos
    results = [[],[]]                    # Variable para almacenar los reseultados de Pangea

    # Pool de procesos
    pool = multiprocessing.Pool(processes=10)
    island(1, solutions, history)
    
    population = solutions
    
    res = pool.map(get_fitness, population)
    for p, f in zip(population, res): 
        if str(p) not in calls: calls[str(p)]=f
    print("Mejor individuo: " + str(get_fittest(population)))
    print("Mejor fitness: " + str(get_fitness(get_fittest(population))))
    print("[X,Y]: " + str(decode(bounds, 32, get_fittest(population))))
    print("Número total de individuos explorados:" + str(evaluaciones))
    print("Tiempo de ejecución (s): " + str(time.time() - start_time))

    # Borrando archivos temporales creados por la paralelización
    for filename in glob.glob("./agent*"):
        os.remove(filename)
        
    # Guardamos los resultados obtenidos
    file='best_found_agent_ga.py'
    with open(file, 'w') as filetowrite:
        filetowrite.write("individual="+str(decode(bounds, 32, get_fittest(population)))+"\n"+program+end_prog)

    minf = [x[0] for x in history]
    avgf = [x[1] for x in history]
    # Plot de los datos resultados obtenidos
    fig, ax=plt.subplots(figsize=(9, 4))
    plt.plot(minf, label='Best individual fitness')
    ax.legend()
    plt.show()
    # Plot de los datos resultados obtenidos
    fig, ax=plt.subplots(figsize=(9, 4))
    plt.plot(avgf, label='Mean fitness')
    ax.legend()
    plt.show()
