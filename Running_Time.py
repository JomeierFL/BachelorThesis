import numpy as np
import secrets
import os
from decimal import Decimal

#Mode: Pareto, Exponential, OneMax or dynamic BinVal? 
mode = "BinVal"

#EA, GA or GAVAR?
algo = "EA"

#Population size
mu = 2

#dimension
N = 3000

#mutation rate, just modify c. It is divided by 100, 
c_ = 100
c = c_/100.0
rate = c/N

#over how many rounds do we average?
numRounds = 30

#iteration limit
limit = 100 * np.exp(c) / c * N * np.log(N)

#parameters for pareto distribution
beta = 0.5
xmin = 1

#parameter for exponential distribution
lam = 0.1

#we change the weights every s rounds
s = 1




#weights for Exponential, Pareto and OneMax
weights = np.zeros(N)

#permutation for dynamic Binval
permutation = np.random.permutation(N)

#global Variables used during the algorithm
rounds = 0 #global variable to count the number of rounds
maxSoFar = 0.0 #Maximum number of 1 bits observed in an individual
currMax = 0.0 #Maximum number of 1 bits in the current population

#initial population
pop = np.random.randint(0,2,(mu,N))

#helper array storing the values of the population according to our function
f = np.zeros(mu)

#two files to write output into
metaFile = open("IOHprofiler_f1_DIM%d_i1_mu%d_c%d_s%d_%s.info" % (N,mu,c_,s,algo), "a")
dataFile = open("IOHprofiler_f1_DIM%d_i1_mu%d_c%d_s%d_%s.dat" % (N,mu,c_,s,algo), "a")


#get a N-dimensional vector with values distributed according to Par(beta,xmin)
def randomParetoWeights():
    global beta
    global xmin

    weights = np.random.pareto(beta,size = N)
    return weights

#get a N-dimensional vector with values distributed according to Exp(lam)
def randomExponentialWeights():
    global lam

    weights = np.random.exponential(lam,size=N)
    return weights

#evaluate f for all individuums in our population, also return the value f(offspring)
def eval(offspring):
    global f
    global mode
    global weights
    global s
    global rounds

    if (rounds-1) % s == 0:
        if mode == "Pareto":
            weights = randomParetoWeights()
        elif mode == "Exponential":
            weights = randomExponentialWeights()
        elif mode == "OneMax":
            weights = np.ones(N)

    f = np.matmul(pop,weights)
    
    return np.dot(offspring,weights)

#compare 2 strings using a permutation of {0,..,N-1} to give weights to the bits, returns 1 if x1 < x2
def less(x1,x2):
    global permutation

    for i in range(N):
        if(x1[permutation[i]] < x2[permutation[i]]):
            return True
        elif (x1[permutation[i]] > x2[permutation[i]]):
                return False

    return True


#create the offspring by mutating the parent, using geometric distribution to make it faster
def mutate(parent):
    global rate

    #bit to be mutated
    _0flip = False
    newString = np.copy(parent)

    mutIndex = 0
    mutIndex = mutIndex + np.random.geometric(rate) - 1
    while mutIndex < N:
        if newString[mutIndex] == 0:
            _0flip = True
        newString[mutIndex] = (newString[mutIndex] + 1) % 2
        mutIndex = mutIndex + np.random.geometric(rate)

    return (newString,_0flip)

#do a crossover
def crossOver(index1,index2):
    global pop

    string1 = pop[index1]
    string2 = pop[index2]
    newString = np.zeros(N)

    for j in range(N):
        flip = np.random.randint(2)
        if flip == 0:
            newString[j] = string1[j]
        else:
            newString[j] = string2[j]

    return newString
    
#update the Population given the offspring
def updatePop(offspring):
    global maxSoFar
    global f
    global mode
    global pop
    global currMax
    global permutation
    global rounds
    global s

    #helper variable to see if population has changed
    accept = False
 
    if mode == "Pareto" or mode == "Exponential" or mode == "OneMax":

        #calculate values for the current population and the weight of the offspring
        offspringWeight = eval(offspring)

        #find index of the worst individual
        min = f[0]
        minIndex = 0

        for i in range(mu):
            if f[i] < min:
                min = f[i]
                minIndex = i

        #if offspring is better than the worst individual, replace
        if(offspringWeight > min):
            accept = True
            pop[minIndex] = offspring

   
    elif mode == "BinVal":

        #pick a random permutation of [0..N-1] if rounds mod s == 0
        if (rounds-1) % s == 0:
            permutation = np.random.permutation(N)

        #find index of the worst individual
        minIndex = 0

        for i in range(mu):
            if less(pop[i],pop[minIndex]) == 1:
                minIndex = i

        #if offspring is better than the worst individual, replace
        if(less(pop[minIndex],offspring)):
            accept = True
            pop[minIndex] = offspring

    #finally, update progress
    if accept:
        maxSoFar = max(maxSoFar, np.sum(offspring))
        currMax = 0
        for i in range(mu):
            currMax = max(currMax, np.sum(pop[i]))


#one round of selection
def _1round():
    global pop
    global mu
    global algo

    if algo == "GA":

        isMut = np.random.randint(2)

        if isMut:
            #determine parent
            index = np.random.randint(0,mu)
            offspring,_0flip = mutate(pop[index])

            if(_0flip or mu > 1):
                updatePop(offspring)
    
        else:
            index1 = np.random.randint(0,mu)
            index2 = np.random.randint(0,mu)

            offspring = crossOver(index1,index2)
            updatePop(offspring)

    elif algo == "GAVar":

        isMut = np.random.randint(2)

        if isMut:
            #determine parent
            index = np.random.randint(0,mu)
            offspring,_0flip = mutate(pop[index])

            if(_0flip or mu > 1):
                updatePop(offspring)
    
        else:
            index1 = np.random.randint(0,mu)
            index2 = np.random.randint(0,mu)

            while index1 == index2:
                index2 = np.random.randint(0,mu)

            offspring = crossOver(index1,index2)
            updatePop(offspring)      

    else:
        #determine parent
        index = np.random.randint(0,mu)
        offspring,_0flip = mutate(pop[index])

        #We only update if a 0-bit was flipped, to improve efficiency
        if(_0flip or mu > 1):
            updatePop(offspring)


#run the evolutionary algorithm once
def optimise(currRound):
    global rounds
    global maxSoFar
    global pop
    global c
    global mu
    global N
    global currMax
    global limit

    #reset rounds
    rounds = 0

    #create initial population, initialize current progress
    pop = np.random.randint(0,2,(mu,N))
    maxSoFar = 0
    for i in range(mu):
        maxSoFar = max(maxSoFar, np.sum(pop[i]))

    currMax = maxSoFar

    #mutate and select until maximum is found, count the rounds
    while maxSoFar < N and rounds < limit:
        rounds = rounds + 1
        _1round()
        dataFile.write("%s +%s +%s\n" % (rounds, f"{Decimal(currMax.item()):.5e}", f"{Decimal(maxSoFar.item()):.5e}"))

#run multiple rounds of an evolutionary algorithm
def test():
    global N
    global mu
    global pop
    global numRounds
    global rate
    global maxSoFar

    #actually run numRounds rounds, storing results in results
    for currRound in range(numRounds):
        dataFile.write('"function_evaluation" "current f(x)" "best-so-far f(x)"\n')
        optimise(currRound)
        metaFile.write("%s%s%s%s, " % ("1:", rounds, "|" , f"{Decimal(maxSoFar.item()):.5e}"))


metaFile = open("IOHprofiler_f1_DIM%d_i1_mu%d_c%d_s%d_%s.info" % (N,mu,c_,s,algo), "a")
dataFile = open("IOHprofiler_f1_DIM%d_i1_mu%d_c%d_s%d_%s.dat" % (N,mu,c_,s,algo), "a")

metaFile.write("suite = 'PBO', funcId = 1, DIM = %d, algId = '%d+1-%s c = %.3f'\n%%\ndata_f1/IOHprofiler_f1_DIM%d_i1_mu%d_c%d_s%d_%s.dat, " % (N,mu,algo,c,s,N,mu,c_,s,algo))

test()

metaFile.close()
dataFile.close()
