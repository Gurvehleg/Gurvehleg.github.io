#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:54:34 2021

@author: lienardr
"""

#%%
from numpy import pi,cos,sin,sqrt,log, log10
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo # Le pro du curve fitting !
import scipy.stats as st
#%% Définition des paramètres


P = 0.5  #Probabilité de tourner 

l = 1 # Longueur de la boite

N = 10000 # Nombre de pas

alpha = np.random.uniform(0,2*pi,N) # Angle pris par la bactérie

l0 = l/100 #longeur d'un pas (il faut prendre un pas l0 fixe de façon à ce que la bactérie explore une distance deux fois plus grand si N*2)

x0,y0 = l/2,l/2 #Positions initiales

N_exp = 100 #Nombre d'iteration

Pas_temps=np.linspace(1,N,N) # Pas de temps


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

#%%  Plot d'une bactérie

xmin,xmax = 0.45,0.55
ymin,ymax = 0.45,0.55


# plt.figure()
plt.grid()
plt.plot(x,y)
#plt.axis('equal') # Image non déformée
#plt.xlim(xmin,xmax)
#plt.ylim(ymin,ymax)
plt.title("Marche aléatoire d'une bactérie une probabilité de tourner P =" +str(P))
plt.xlabel("Pas verticaux")
plt.ylabel("Pas horizontaux")
plt.plot(x0,y0, marker="o", color="red", label="début") #Trace un point pour l'origine de départ de la bactérie
plt.plot(x[-1],y[-1], marker="o", color="violet", label="fin") #Trace un point pour l'origine de départ de la bactérie
plt.legend()
plt.show()


#%%
#%% Fonction 

def marche_bacterie(N,P):
    
    
    """
    Fonction de marche aléatoire d'une bactérie en l'absence d'interaction et de gradient
    """
    
    
    x=np.zeros(N) #Matrice des positions en x et en y
    y=np.zeros(N)
    
    alpha = np.random.uniform(0,2*pi,N) # Angle pris par la bactérie lorsqu'elle tourne
    changes = np.random.binomial(1,P,N) #Determine si on tourne (1) ou pas (0)

    
    
    #Conditions initiales
    x[0]=x0 
    y[0]=y0
    changes[1]=1 #On doit initialement faire tourner la bactérie car elle n'a pas de direction à "poursuivre" à la premiere itération
    
    for i in range (0,N-1) :
        if changes[i+1] == 1 : #La bactérie tourne d'un angle alpha 
            x[i+1] = l0*cos(alpha[i+1]) + x[i]
            y[i+1] = l0*sin(alpha[i+1]) + y[i]
        else :
            alpha[i+1]=alpha[i]  # On doit "fixer la mémoire" en particulier si on ne tourne pas deux fois de suite
            x[i+1] = l0*cos(alpha[i+1]) + x[i]
            y[i+1] = l0*sin(alpha[i+1]) + y[i]
    return x,y
#%% Calcul du MSD

temp = np.zeros((N,N_exp)) #Fonction MSD mais sans la moyenne (ligne,colonne)
MSD = np.zeros(N)

"""
for j in range (0,N_exp) : #On modifie chaque colonne j par les données d'une expérience
    x,y=marche_bacterie(N,P)[0]-x0, marche_bacterie(N,P)[1]-y0 #On appelle la fonction qui nous sort un jeu de variable x,y pour la trajectoire d'une bactérie
    temp[:,j]  = x[:]**2 + y[:]**2 #On regarde la colonne j et on y injecte nos valeurs
    
    

Ici on a calculé le MSD pour chaque ligne avec x0 <r(t)^2> = < (x(t-t0)-x(t0))^2+(y(t-t0)-y(t0))^2 >, on  a fixé t0 au premier temps et t
 correspond ensuite au temps 1,2,3 mais on doit calculer un MSD pour chaque ligne de la matrice, et pour chaque expérience. 
 Le fait de faire plusieurs expériences nous permet de réduire l'incertitude sur le MSD mais on aurait aussi pu refaire un calcul 
 en prenant un MSD a un t0 différent (par exemple t=2). On aurait pu comme ça augmenter artificeillement notre nombre de valeurs mais 
 nos incertitudes auraient été corrélées"
 
"""
#Version optimisée :
    
for j in range (0,N_exp) : 
    exp = marche_bacterie(N,P)
    temp[:,j]  = (exp[0]-x0)**2 + (exp[1]-y0)**2 
    
#%% Fonction MSD
    
def MSD (N_experience) :
    """
    
    N_experience = nombre d'expérience, ici le nombre de colonnes dans la matrice
    Nombre_pas = Nombre de pas de la bactérie
    
    
    """

    temp = np.zeros((N,N_exp)) #Fonction MSD mais sans la moyenne (ligne,colonne)
    MSD = np.zeros(N)

    for j in range (0,N_exp) : 
        exp = marche_bacterie(N,P)
        temp[:,j]  = (exp[0]-x0)**2 + (exp[1]-y0)**2 
        
    for i in range(0,N) :
        MSD[i]=np.mean(temp[i,:]) 
        
    return MSD
#%% Calcul incertitudes
    
    
#t_student=1.6 #A calculer    
Incertitudes = np.zeros(N)
sigma=np.zeros(N)   
    
for i in range(0,N) :
    
    MSD[i]=np.mean(temp[i,:])   #On calcule MSD pour le premier temps 
    sigma[i]=np.std(temp[i,:],ddof=1)  #Estimateur en n-1

#%%
t_student=st.t.interval(0.95,(N_exp)-1) #intervalle de confiance, nombre de paramètres libres. Ici on a fixé la moyenne donc n-1
 
Incertitudes= sigma*t_student[1]/sqrt(N_exp)



#%% plot du MSD
plt.figure()
plt.loglog(Pas_temps,MSD)
plt.errorbar(Pas_temps,MSD, yerr=Incertitudes)
plt.title("MSD en fonction du temps sur "+str(N_exp)+" expériences, P="+str(P))
plt.grid()
plt.xlabel("Temps (en nombre de pas)")
plt.ylabel("MSD")
plt.show()
#%% Ajustement MSD à grand N (temps)

abscisse = Pas_temps[N//1000:N]  #On prend de N//100 à N valeurs
ordonnee = MSD[N//1000:N]
y_erreur = Incertitudes[N//1000:N]/(ordonnee+log(10))  #Simplement l'erreur d'un logarithme base 10 attention
x_erreur=np.zeros(N-N//1000) # Pas d'erreur sur x




"incertitude statisque entre les n mesures mean et std avec coef student" 
#%% Curve fitting 
x = log10(abscisse)
y = log10(ordonnee)

p0 = np.array([1.5,10]) # initial guess

uy = y_erreur
ux = x_erreur

def f0(x,a,b): 
    return a*x+b

popt,pcov=spo.curve_fit(f0,x,y,sigma=uy,absolute_sigma=True)


upopt = np.sqrt(np.abs(np.diagonal(pcov)))

print('pente = ' + str(round(popt[0],4)) + ' ± ' + str(round(upopt[0],4)))
print('ordonnée à l\'origine = ' + str(round(popt[1],9)) + ' ± ' + str(round(upopt[1],9)))


#%% Plot curve fit

plt.figure()
x_mod = np.linspace(np.amin(x),np.amax(x),1000)
y_mod = f(x_mod,popt[0],popt[1])
plt.plot(x_mod,y_mod,label='Modèle affine $y = a x + b$', color = 'blue', linestyle= '-')
plt.plot(x,y)
plt.errorbar(x,y,xerr=ux,yerr=uy,marker='+', color = 'red', linestyle= '',label='Simulation')
#plt.errorbar(x_exp,y_exp,xerr=ux_exp,yerr=uy_exp,marker='+', color = 'green', linestyle= '',label='Point(s) pris en direct')
# plt.xlabel('ln(t)',fontsize=25)
# plt.ylabel('ln(MSD))',fontsize=25)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# #plt.xlim(0.9*np.amin(x), 1.01*np.amax(x))
#plt.ylim(0.9*np.amin(y), 1.01*np.amax(y))
plt.legend(loc='upper left',fontsize=15)
plt.title("Ajustement du MSD pour une probabilité de P="+str(P))
# plt.text((np.amin(x)+np.amax(x))/2.5, np.amin(y) + (np.amin(y) + np.amax(y))/10 , 'Pente a = ' + str(round(popt[0],15)) + ' ± ' + str(round(upopt[0],10)) + ' unité',fontsize=16)
# plt.text((np.amin(x)+np.amax(x))/2.5, np.amin(y) + (np.amin(y) + np.amax(y))/20, 'Ordonnée à l\'origine = ' + str(round(popt[1],10)) + ' ± ' + str(round(upopt[1],10)) + ' unité',fontsize=16)
plt.show()


#%% 

def test (a):
    x = np.random.binomial(1,0.5,10)
    y=x
    return x,y
#%%

a=test(4)
print(a)

e,f = test (a)[0], test (a) [1]
#%%
#%%
#%%
#%%
#%%
#%%
#%%
