import numpy as np
from scipy.special import binom
from scipy import stats
import matplotlib.pyplot as plt

#Mode: Pareto or BinVal?
mode = "BinVal"

#EA, GA or GAVAR?
algo = "EA"

#Population size
mu = 2

#dimension
N = 3000

#mutation rate
c = 4.2
rate = c/N

#ranges for inital ones
startOne = 2615
endOne = 3000
stepOne = 20

#over how many rounds do we average?
numRounds = 200000

#parameters for pareto distribution
beta = 0.5
xmin = 1

#parameter for exponential distribution
lam = 0.1

#global variable to count rounds
rounds = 0

#current Maximum number of 1 bits in an individual
currMax = 0

#initial population
pop = np.random.randint(0,2,(mu,N))

#helper array storing the values of the population according to our function
f = np.zeros(mu)

#file to store results
summary = open("Drift_N%d_c%.3f_%s_(%d + 1)-%s_summary.txt" % (N,c, mode, mu, algo), "a")
data = open("Drift_N%d_c%.3f_%s_(%d + 1)-%s_data.txt" % (N,c, mode, mu, algo), "a")
plotMeans = open("Drift_N%d_c%.3f_%s_(%d + 1)-%s_plotMeans.txt" % (N,c, mode, mu, algo), "a")
plotOnes = open("Drift_N%d_c%.3f_%s_(%d + 1)-%splotOnes.txt" % (N,c, mode, mu, algo), "a")
plotErr = open("Drift_N%d_c%.3f_%s_(%d + 1)-%s_plotErr.txt" % (N,c, mode, mu, algo), "a")




#check if all Individuums of the population are equal
def allEqual():
    global pop
    global mu

    for i in range(mu):
        if not np.array_equal(pop[0],pop[i]):
            return False
    
    return True


#create a random individuum of length N with NumZeros 0-bits.
def createInd(NumZeros):
    global N

    individuum = np.zeros(N)
    for i in range(NumZeros, N, 1):
        individuum[i] = 1
    
    return individuum


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

    if mode == "Pareto":
        weights = randomParetoWeights()
    elif mode == "Exponential":
        weights = randomExponentialWeights()
    elif mode == "OneMax":
        weights = np.ones(N)

    f = np.matmul(pop,weights)
    
    return np.dot(offspring,weights)

#compare 2 strings using a permutation of {0,..,N-1} to give weights to the bits, returns 1 if x1 < x2
def less(x1,x2,perm):

    for i in range(N):
        if(x1[perm[i]] < x2[perm[i]]):
            return True
        elif (x1[perm[i]] > x2[perm[i]]):
                return False

    return True


#create the offspring by mutating the parent, using geometric distribution to make it faster
def mutate(parent,y):
    global rate

    #bit to be mutated
    _0flip = False
    newString = np.copy(parent)

    mutIndex = y
    mutIndex = mutIndex + np.random.geometric(rate) - 1
    while mutIndex < N:
        if newString[mutIndex] == 0:
            _0flip = True
        newString[mutIndex] = (newString[mutIndex] + 1) % 2
        mutIndex = mutIndex + np.random.geometric(rate)

    return (newString,_0flip)

#do a crossover between two strings
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
    global currMax
    global f
    global mode
    global pop
 
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
            pop[minIndex] = offspring

   
    elif mode == "BinVal":

        #pick a random permutation of [0..N-1]
        permutation = np.random.permutation(N)

        #find index of the worst individual
        minIndex = 0

        for i in range(mu):
            if less(pop[i],pop[minIndex],permutation) == 1:
                minIndex = i

        #if offspring is better than the worst individual, replace
        if(less(pop[minIndex],offspring,permutation)):
            pop[minIndex] = offspring

    #finally, update progress
    currMax = max(currMax, np.sum(offspring))


#one round of selection
def _1round(initPop, y):
    global pop
    global mu
    global algo
    global c
    global N
    global currMax

    #As long as we have the initial population, always flip a 1-bit
    if initPop:
    
        #Probi is the array to store prefix sum of the probabilites to get 1,2,..,y ones conditioned that we get at least 1
        Probi = np.zeros(y)

        #Probablity to get no ones flipped
        Prob0 = 1 - np.power(1-c/N,y)

        #initialize the first entry
        Probi[0] = y * (c/N) * np.power(1-c/N,y-1) / Prob0

        #Calculate every other entry
        for i in range(1,y,1):
            Probi[i] = Probi[i-1] + (binom(y,i+1) * np.power(c/N,i+1) * np.power(1-c/N,y-i-1) / Prob0)

        #now draw random number between 0 and 1 to see how many ones are flipped
        r = np.random.random_sample()
        for i in range(y):
            if r < Probi[i]:
                #how many 1 bits are flipped?
                num1flips = i+1
                break

        #mutate the bits b[y] to b[N] normally
        offspring, _0flip = mutate(pop[0],y)

        #flip the first num1flips 1-bits
        for i in range(num1flips):
            offspring[i] = 1


        updatePop(offspring)

    



    else:

        if algo == "GA":
            isMut = np.random.randint(2)

            if isMut:
                #determine parent
                index = np.random.randint(0,mu)
                offspring,_0flip = mutate(pop[index],0)

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
                offspring,_0flip = mutate(pop[index],0)

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
            offspring,_0flip = mutate(pop[index],0)

            if(_0flip or mu > 1):
                updatePop(offspring)

            
#start with mu identical individuals with k ones and iterate until the population is degenerated. Repeat numRounds times.
def degenerate(k):
    global numRounds
    global pop
    global c
    global N
    global mu
    global summary
    global data
    global rounds
    global plotMeans
    global plotErr
    global plotOnes

    #results[100] is the number of degenerated pops with the same number of 1s, then results[99] is the number with one "1" less etc.
    results = np.zeros(200)

    for _ in range(numRounds):
        ind = createInd(N-k)
        for i in range(mu):
            pop[i] = ind
        
        helper = False

        if mu > 1:

            while(not allEqual() or (not helper)):
                rounds = rounds + 1
                if not helper:
                    _1round(True,N-k)
                else:
                    _1round(False,0)

                if not allEqual() and (not helper):
                    helper = True

        else:

            _1round(False,0)
            rounds = rounds + 1
        
        data.write("%d\n" % (np.sum(pop[0]) - k))
        results[np.sum(pop[0]) - k + 100] = results[np.sum(pop[0]) - k + 100] + 1
           
    
    #some statistics, we do a t-test to see if the drift is "truly" negative/positive.
    print("Started with", k , "ones, c = ",c, "N =", N)
    mean = sum([((x-100) * results[x]/np.sum(results)) for x in range(200)])
    S = np.sqrt(sum([results[x] * np.square((x-100)-mean) for x in range(200)]) / (numRounds-1)) 
    T = np.sqrt(numRounds) * mean / S
    print("Expected value:", mean)
    if mean > 0:
        tmp = 1 - stats.t.cdf(T,numRounds-1)
    else:
        tmp = stats.t.cdf(T,numRounds-1)
    print("Probability to observe a more extreme value under H0:" , tmp)
    summary.write(" mean: %f, standard deviation: %f, p-value: %f\n" % (mean, S/np.sqrt(numRounds), tmp))
    plotMeans.write(", %f" % (mean))
    plotErr.write(", %f" % (S/np.sqrt(numRounds)))
    plotOnes.write(", %d" % (k))
    return mean, S


#run degenerate for different number of 1-bits.
def test():
    global startOne
    global endOne
    global stepOne
    global N
    global c
    global mode
    global algo
    global mu
    global numRounds

    x = list()
    y = list()
    error = list()

    for currOnes in range(startOne, endOne, stepOne):
        summary.write("\nOnes: %d" % currOnes)
        data.write("\nOnes: %d\n" % currOnes)
        mean, sigma = degenerate(currOnes)
        error.append(sigma)
        y.append(mean)
        x.append(currOnes)

    plt.plot(x, y, 'k', color='#CC4F1B')
    plt.fill_between(x, [i-(j/np.sqrt(numRounds)) for i,j in zip (y, error)], [i+(j/np.sqrt(numRounds)) for i,j in zip (y, error)],alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig("N%d_c%.2f_%s_(%d + 1)-%s.png" % (N,c,mode,mu,algo))


test()

summary.write("numRounds: \n", numRounds)
data.close()
summary.close()
plotOnes.close()
plotMeans.close()
plotErr.close()