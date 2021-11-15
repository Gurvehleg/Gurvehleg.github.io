#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:54:34 2021

@author: lienardr
"""

#%%
from numpy import pi,cos,sin
import numpy as np
import matplotlib.pyplot as plt
#%% Définition des paramètres


P = 0.8  #Probabilité de tourner 

l = 1 # Longueur de la boite

N = 1000 # Nombre de pas

alpha = np.random.uniform(0,2*pi,N) # Angle pris par la bactérie

l0 = l/N #longeur d'un pas

x0,y0 = l/2,l/2




#%%

x=np.zeros(N)
y=np.zeros(N)

changes = np.random.binomial(1,P,N) #Determine si on tourne (1) ou pas (0)
#changes[0]=1


x[0]=x0
y[0]=y0


for i in range (0,N-1) :
    #print(alpha[i])
    if changes[i+1] == 1 :
        x[i+1] = l0*cos(alpha[i+1]) + x[i]
        y[i+1] = l0*sin(alpha[i+1]) + y[i]
    else :
        alpha[i]=alpha[i-1]  # On doit "fixer la mémoire" en particulier si on tourne deux fois de suite
        x[i+1] = l0*cos(alpha[i]) + x[i]
        y[i+1] = l0*sin(alpha[i]) + y[i]

#%% Figure

xmin,xmax = 0.45,0.55
ymin,ymax = 0.45,0.55


# plt.figure()
plt.grid()
plt.plot(x,y)
# plt.axis('equal') # Image non déformée
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("Marche aléatoire d'une bactérie")
plt.xlabel("Pas verticaux")
plt.ylabel("Pas horizontaux")
plt.plot(x0,y0, marker="o", color="red", label="début") #Trace un point pour l'origine de départ de la bactérie
plt.plot(x[-1],y[-1], marker="o", color="violet", label="fin") #Trace un point pour l'origine de départ de la bactérie
plt.legend()
plt.show()

#%% Boucle d'éxécution

alpha=[]

for i in range (0,N) :
    alpha.append(np.random.uniform(0,2*pi))

cos(pi)
#%%
#%% Test
print(alpha)
print(len(alpha))
u = np.random.uniform(0,2*pi)
#%%

#%%
#%%

