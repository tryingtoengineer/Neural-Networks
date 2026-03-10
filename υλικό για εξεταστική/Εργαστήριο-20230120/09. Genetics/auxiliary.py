
###########################################################################################
########### computing #####################################################################
###########################################################################################
import numpy as np
from numpy import linspace, sin, pi, array, zeros, hstack, cumsum
from numpy import vectorize, ones
from numpy.random import seed, random, randint

# Random generates n numbers between xa and xb
def Random(n, xa, xb): return random(n) * (xb - xa) + xa

# Flip generates a Bernoulli variable; throw a coin with probability p
def FlipCoin(p):
    if p==1.0: return True
    if p==0.0: return False
    if random()<=p: return True
    return False

# SimpleChromo splits x into n unequal parts
def SimpleChromo(x, n):
    vals = random(n)
    sumv = sum(vals)
    return x * vals / sumv

# Fitness function: map objective function into [0, 1]
#  Y -- objective values
def Fitness(Y):
    ymin, ymax = min(Y), max(Y)
    if abs(ymax - ymin) < 1e-14: return ones(len(Y))
    return (ymax - Y) / (ymax - ymin)


# SortPop sorts individuals by fitness (decreasing order)
#  C -- chromosomes/population
#  Y -- objective values
#  F -- fitness
def SortPop(C, Y, F):
    I = F.argsort()[::-1] # the [::-1] is a trick to reverse the sorting order
    C = C[I]              # sorted chromosomes
    Y = Y[I]              # sorted objective values
    F = F[I]              # sorted fitness
    return C, Y, F

# PrintPop prints all individuals
#  C    -- chromosomes/population
#  Y    -- objective values
#  xFcn -- converts C to X values
#  F    -- fitness
#  P    -- probabilities
#  M    -- cumulated probabilities
def PrintPop(C, Y, xFcn, F=None, P=None, M=None, showC=False):
    print('%7s%7s' % ('x', 'y'), end = '')
    X = array([xFcn(c) for c in C])
    if showC:
        L = []
        for c in C:
            l = ''
            for v in c: l += '%7.3f' % v
            L.append(l)
        nc = str(len(C[0]) * 7)
        print(('%'+nc+'s') % ('chromosome/genes'), end = '')
    if np.any(F)!=None: print('%8s' % 'fitness', end = '')
    if np.any(P)!=None: print('%8s' % 'prob', end = '')
    if np.any(M)!=None: print('%8s' % 'cum.prob', end = '')
    print()
    for i, x in enumerate(X):
        print ('%7.2f%7.2f' % (x, Y[i]), end = '')
        if showC: print (L[i], end = '')
        if np.any(F)!=None: print('%8.3f' % F[i], end = '')
        if np.any(P)!=None: print('%8.3f' % P[i], end = '')
        if np.any(M)!=None: print('%8.3f' % M[i], end = '')
        print()
        
# RouletteSelect selects n individuals
#  M -- cumulated probabilities
def RouletteSelect(M, n, sample=None):
    if np.any(sample)==None: sample = random(n)
    S = zeros(n, dtype=int) # selected individuals
    for i, s in enumerate(sample):
        for j, m in enumerate(M):
            if m > s:
                S[i] = j
                break
    return S

# FilterPairs generates 2 x ninds/2 lists from selected individuals
# Try to avoid repeated indices in pairs
def FilterPairs(S):
    ninds = len(S)
    A = zeros(ninds//2, dtype=int)
    B = zeros(ninds//2, dtype=int)
    for i in range(ninds//2):
        a, b = S[2*i], S[2*i+1]
        if a == b:
            for s in S:
                if s != a:
                    b = s
                    break
        A[i], B[i] = a, b
    return A, B

###########################################################################################
########### plotting ######################################################################
###########################################################################################

from matplotlib.patches import Rectangle, FancyArrowPatch
from pylab import grid, xlabel, ylabel, legend, plot, show, close, subplot
from pylab import clf, gca, xticks, text, axis, savefig, rcParams

# Gll adds grid, labels and legend
def Gll(xl, yl, legpos=None): grid(); xlabel(xl); ylabel(yl); legend(loc=legpos)

# PlotProbBins plots probabilities bins
#  X -- population
#  P -- probabilities
def PlotProbBins(X, P):
    rcParams.update({'figure.figsize':[800/72.27,200/72.27]})
    x0, Tk = 0.0, [0.0]
    for i in range(len(X)):
        gca().add_patch(Rectangle([x0, 0], P[i], 0.2, color='#d5e7ed', ec='black', clip_on=0))
        ha = 'center'
        if i==len(X)-1: ha = 'left' # last one
        text(x0+P[i]/2.0, 0.1, '%.1f'%X[i], ha=ha)
        x0 += P[i]
        Tk.append(x0)
    xticks(Tk, ['%.2f'%v for v in Tk])
    axis('equal')
    gca().get_yaxis().set_visible(False)
    for dir in ['left', 'right', 'top']:
        gca().spines[dir].set_visible(False)
    xlabel('cumulated probability')
    grid()
    axis([0, 1, 0, 0.2])

# DrawChromo draws one chromosome
def DrawChromo(key, A, pos, y0, swap_colors, red='#e3a9a9', blue='#c8d0e3'):
    nbases = len(A)
    x0, l = 0.1, 1.0 / float(nbases)
    red, blue = red, blue
    text(x0-0.01, y0+0.05, key, ha='right')
    if swap_colors: red, blue = blue, red
    for i in range(0, pos):
        gca().add_patch(Rectangle([x0, y0], l, 0.1, color=red, ec='black'))
        text(x0+l/2.0, y0+0.05, '%.3f'%A[i], ha='center')
        x0 += l
    for i in range(pos, nbases):
        gca().add_patch(Rectangle([x0, y0], l, 0.1, color=blue, ec='black'))
        text(x0+l/2.0, y0+0.05, '%.3f'%A[i], ha='center')
        x0 += l

# DrawCrossover draws crossover process
def DrawCrossover(A, B, a, b, pos):
    rcParams.update({'figure.figsize':[800/72.27,400/72.27]})
    DrawChromo('A', A, pos, 0.35, 0)
    DrawChromo('B', B, pos, 0.25, 1)
    DrawChromo('a', a, pos, 0.10, 0, blue='#e3a9a9')
    DrawChromo('b', b, pos, 0.00, 0, red='#c8d0e3')
    axis('equal')
    axis([0, 1.2, 0, 0.4])
    gca().get_yaxis().set_visible(False)
    gca().get_xaxis().set_visible(False)
    for dir in ['left', 'right', 'top', 'bottom']:
        gca().spines[dir].set_visible(False)
    gca().add_patch(FancyArrowPatch([0.6,0.25], [0.6, 0.2], fc='#9fffde', mutation_scale=30))