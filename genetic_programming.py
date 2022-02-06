from py4j.java_gateway import JavaGateway
import matplotlib.pyplot as plt
import random
import operator
import csv
import itertools
import multiprocessing
import numpy
import math
import os, glob
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

gateway = JavaGateway()  

program = "from scipy import interpolate\nimport sys\nimport numpy\nfrom math import *\nfrom operator import *\ndef if_then_else(input, output1, output2):\n\tif input: return output1\n\telse: return output2\ndistance = 0\nprev = 0\nt = 0\nimport random\ncheckpoints = int (input())\ntargets = []\ndef getAngle(a, b, c):\n\tang = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))\n\treturn ang + 360 if ang < 0 else ang\ndef protectedDiv(left, right):\n\ttry: return left / right\n\texcept ZeroDivisionError: return 1\ndef fitness(x, y, index):\n\tglobal t\n\ttotal = dist((x,y), targets[index])\n\tfor i in range(index+1, checkpoints-1):\n\t\ttotal += dist(targets[i], targets[i+1])\n\treturn total + t\ndef bound(low, high, value):\n\treturn max(low, min(high, value))\nfor i in range (checkpoints) :\n\ttargets += [[int (j) for j in input(). split()]]\ntargets = targets * 2\nwhile True:\n\tcheckpoint_index, x, y, vx, vy, angle = [int(i) for i in input().split()]\n\tprint('['+str(fitness(x,y,checkpoint_index))+','+str(checkpoint_index)+','+str(x)+','+str(y)+','+str(vx)+','+str(vy)+','+str(angle)+']', file = sys.stderr )\n\tif dist((x,y), targets[checkpoint_index]) > 600:\n\t\tctr =numpy.array([[x, y]] + targets[checkpoint_index:checkpoint_index+4]) \n\t\tx_=ctr[:,0] \n\t\ty_=ctr[:,1] \n\t\ttck,u = interpolate.splprep([x_,y_],k=2,s=0)\n\t\tu=numpy.linspace(0,1,num=30,endpoint=True)\n\t\tout = interpolate.splev(u,tck)\n\t\tX = out[0][1]\n\t\tY = out[1][1]\n\telse:\n\t\tX = vx\n\t\tY = vy\n\tfriction = 0.15\n\tmax_angle = radians(18)\n\tmax_thrust = 200\n\tangle = radians(getAngle((vx, vy),(x,y),(X,Y)))\n\tt+=1\n\t"
end_prog = "\n\tprint(str(int(X)) + ' '+ str(int(Y)) + ' ' +str(int(thrust))+ ' agent')"

# Definición de nuevas funciones
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Operadores lógicos
# Definición de una nueva función if-then-else
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

# Conjunto de primitivas
pset = gp.PrimitiveSetTyped("main", [float, float, float, float, float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# Conjunto de terminales
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)
pset.renameArguments(ARG0='friction', ARG1='max_angle', ARG2='max_thrust', ARG3='angle',ARG4='vx', ARG5='vy')

# Definimos un problema de minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Mitad de lapoblación inicializada por método grow y la otra mitad por método full
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Función de evaluación
# - Puede cambiarse el número de mapas a probar si se requiere
def eval_individual(individual, points):
    pid = str(os.getpid()) + '.py'
    func = toolbox.compile(expr=individual)
    file='agent'+pid
    with open(file, 'w') as filetowrite:
        filetowrite.write(program+"thrust=abs(bound(-200,200,"+str(individual)+'))'+end_prog)
    
    fitness = 0

    result = gateway.entry_point.simulate(3, pid)  # invoke constructor
    result = next((dop for dop in reversed(result['0']) if dop is not None), None)
    fitness += eval(result)[0]

    result = gateway.entry_point.simulate(4, pid)  # invoke constructor
    result = next((dop for dop in reversed(result['0']) if dop is not None), None)
    fitness += eval(result)[0]
    
    return (float)(fitness/2),

# Definición de los operadores genéticos
toolbox.register("evaluate", eval_individual, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)                        # Fijamos semilla para experimentación
    e = 0.025                               # Elitismo
    pop = toolbox.population(n=10)         # Población de n=100 individuos
    hof = tools.HallOfFame(int(1))      # Tamaño del salón de la fama
    
    # Pool de procesos
    pool = multiprocessing.Pool(processes=10)
    toolbox.register("map", pool.map)

    # Definición las estadísticas recogidas
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Algoritmo genético canónico
    # pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats,
    #                                halloffame=hof, verbose=True)
    
    # Algoritmo Harm
    pop, log = gp.harm(pop, toolbox, 0.5, 0.1, 10, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats, halloffame=hof, verbose=True)
    
    print(hof[0])   # Print del mejor individuo
    
    return pop, log, hof

if __name__ == "__main__":
    pop, logbook, hof = main()
    
    # Borrando archivos temporales creados por la paralelización
    for filename in glob.glob("./agent*"):
        os.remove(filename)
    
    # Guardamos en fichero el mejor indivduo
    file='best_found_agent_gp.py'
    with open(file, 'w') as filetowrite:
        filetowrite.write(program+"thrust=abs(bound(-200,200,"+str(hof[0])+'))'+end_prog)
    
    # Gráfica de la evolución
    gen = logbook.select("gen")
    fit_mins    = logbook.chapters["fitness"].select("min")
    fit_avg     = logbook.chapters["fitness"].select("avg")
    size_avgs   = logbook.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    # line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    line3 = ax1.plot(gen, fit_avg, "g-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line2 + line3 # + line1
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    # Guardado de los resultados y estadísticas relativas al fitness y al tamaño
    df_log = pd.DataFrame(logbook.chapters["fitness"])
    df_log.to_csv('./resultados/fitnesses.csv', index=True)
    df_log = pd.DataFrame(logbook.chapters["size"])
    df_log.to_csv('./resultados/sizes.csv', index=True)
    plt.show()
