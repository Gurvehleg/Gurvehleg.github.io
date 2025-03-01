# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 21:34:18 2021

@author: Rémy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1

@author: lienardr
"""

#%%
from numpy import pi,cos,sin,sqrt,log
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo # Le pro du curve fitting !
import scipy.stats as st

# from time import sleep
from tqdm import tqdm #Progress bar ;)

#%% Définition des paramètres


P = 0.05  #Probabilité de tourner 

l = 1 # Longueur de la boite

N = 1000 # Nombre de pas

alpha = np.random.uniform(0,2*pi,N) # Angle pris par la bactérie

l0 = l/100 #longeur d'un pas (il faut prendre un pas l0 fixe de façon à ce que la bactérie explore une distance deux fois plus grand si N*2)

x0,y0 = l/2,l/2 #Positions initiales

N_exp = 100 #Nombre d'iteration

Pas_temps=np.linspace(1,N,N) # Pas de temps


#%% Fonction marche d'une bactérie

def marche_bacterie(N,P):
    
    
    """
    Fonction de marche aléatoire d'une bactérie en l'absence d'interaction et de gradient
    N = Nombre de pas de la bactérie
    P = Probabilité de tourner
    
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

#%% Fonction MSD avec écart type et incertitudes
    
def MSD (N,N_experience) :
    """
    
    N_experience = nombre d'expérience, ici le nombre de colonnes dans la matrice
    N = Nombre de pas de la bactérie
    
    
    """

    temp = np.zeros((N,N_exp)) #Fonction MSD mais sans la moyenne (ligne,colonne)
    MSD = np.zeros(N)
    sigma=np.zeros(N)   
    Incertitudes = np.zeros(N)
    
    
    for j in tqdm(range (0,N_exp) ): 
        exp = marche_bacterie(N,P) #On récupères les coordonnées d'une expérience j 
        temp[:,j]  = (exp[0]-x0)**2 + (exp[1]-y0)**2 #On calcule le SD
        
    for i in tqdm(range(0,N)) :
        MSD[i]=np.mean(temp[i,:]) # Calcule le M_SD 
        sigma[i]=np.std(temp[i,:],ddof=1)  # Calcule l'écart type avec estimateur en n-1
        
    t_student=st.t.interval(0.95,(N_experience)-1)
    Incertitudes= sigma*t_student[1]/sqrt(N_exp)
    return MSD,sigma,Incertitudes


#%% Distribution Gaussienne de la concentration en nutriments

l = 1 # Longueur de la boite

ecart_type = l/2


C0= 1

def Concentration (x,y,x0,y0) :
    """
    centree en x0,y0
    """
    
    return np.exp(- ((x-x0)**2 +(y-y0)**2) / (2*ecart_type**2))


#%% Mise a jour de la fonction marche_bactérie en prenant en compte le gradiaet 
def marche_bacterie_concentration(N):
    
    
    """
    Fonction de marche aléatoire d'une bactérie en présence de gradient de c0, P proba d'aller tout droit
    """
    
    
    x=np.zeros(N) #Matrice des positions en x et en y
    y=np.zeros(N)
    
    alpha = np.random.uniform(0,2*pi,N) # Angle pris par la bactérie lorsqu'elle tourne
    #changes = np.random.binomial(1,P) #Determine si on tourne (0) ou pas (1) P : probabilité pour la bactérie de continuer en LD

    
    
    #Conditions initiales
    x[0]=np.random.uniform(-l/2,l/2)
    y[0]=np.random.uniform(-l/2,l/2)
    P=0 #On doit initialement faire tourner la bactérie car elle n'a pas de direction à "poursuivre" à la premiere itération
    
    for i in range (0,N-1) :
        changes=np.random.binomial(1,P)
        if changes == 0 : #La bactérie tourne d'un angle alpha, donc elle sent son environnement 
            x[i+1] = l0*cos(alpha[i+1]) + x[i]
            y[i+1] = l0*sin(alpha[i+1]) + y[i]
            #P=Concentration((x[i+1],y[i+1]))
        else : 
            alpha[i+1]=alpha[i]  # On doit "fixer la mémoire" en particulier si on ne tourne pas deux fois de suite
            x[i+1] = l0*cos(alpha[i+1]) + x[i]
            y[i+1] = l0*sin(alpha[i+1]) + y[i]
        P=sqrt(Concentration(x[i+1],y[i+1],x[0],y[0]))
#        P=(Concentration(x[i+1],y[i+1],x[0],y[0]))
    P_fin_de_run= changes #On garde en mémoire si la bactérie est en tubble ou en run
    return x,y,P_fin_de_run



#%% Execution des fonctions :



experience=MSD(N,N_exp)

MSD=experience[0]
sigma=experience[1]
Incertitudes=experience[2]


#%% plot du MSD





plt.figure()
plt.loglog(Pas_temps,MSD)
plt.errorbar(Pas_temps,MSD, yerr=Incertitudes)
plt.title("MSD en fonction du temps sur "+str(N_exp)+" expériences, P="+str(P))
plt.grid()
plt.xlabel("Temps (en nombre de pas)")
plt.ylabel("MSD")
plt.show()


"""
Comportement ballistique au début N<100/10. Aux petites echelles de temps, pas de marche aléatoire pure. On observe le cpt diffusif
aux t grands

On ne peut pas faire confiance aux be car les points ne sont pas statistiquement indépendants
(position[0]-x0)**2 

"""

#%% Ajustement MSD à grand N (temps)

abscisse = Pas_temps[N//100:N]  #On prend de N//100 à N valeurs
ordonnee = MSD[N//100:N]
y_erreur = Incertitudes[N//100:N]/(ordonnee)  #Simplement l'erreur d'un logarithme base 10 attention
x_erreur=np.zeros(N-N//100) # Pas d'erreur sur x


"incertitude statisque entre les n mesures mean et std avec coef student" 
#%% Curve fitting pour le MSD
x = log(abscisse)
y = log(ordonnee)

p0 = np.array([1.5 ,10]) # initial guess

uy = y_erreur
ux = x_erreur

def f0(x,a,b): 
    return a*x+b

popt,pcov=spo.curve_fit(f0,x,y,sigma=uy,absolute_sigma=True)
 

upopt = np.sqrt(np.abs(np.diagonal(pcov)))

print('pente = ' + str(round(popt[0],4)) + ' ± ' + str(round(upopt[0],4)))
print('ordonnée à l\'origine = ' + str(round(popt[1],9)) + ' ± ' + str(round(upopt[1],9)))

#%% Représentation Gradient

Ne=1000# Discrétisation espace


x,y=np.linspace(-l,l,Ne),np.linspace(-l,l,Ne)
M=np.zeros((Ne,Ne))# Position Ne : abscisse l.

for j in range(Ne):
    M[:,j]=Concentration(x[:],y[j],x0=0,y0=0)


#plt.figure()
plt.imshow(M,extent=[-l,l,-l,l])
#plt.plot(x,y)
plt.colorbar(label='Concentration')
#plt.xticks([i*Ne//10 for i in range(10)],labels=np.round(x[::Ne//10],2))#On ne met que 10 ticks
#plt.yticks([i*Ne//10 for i in range(10)],labels=np.round(y[::-Ne//10],2))# - pour avoir le 0 au bon endroit
plt.title('Profil de concentration dans la boîte : gausienne de largeur l/10',fontsize=18)
plt.xlabel('Position dans la boîte (en longueur de boîte)',fontsize=15)
plt.ylabel('Position dans la boîte (en longueur de boîte)',fontsize=15)
# plt.legend(fontsize=15)
#plt.grid(False)
plt.show()


#%% Tracé pour range bactéries

test=[]

for i in range(0,500) :
    
    exp = marche_bacterie_concentration(N)
    
    
    
    # plt.figure()
    #plt.grid()
    
    
    plt.imshow(M,extent=[-l,l,-l,l])
#    plt.plot(exp[0],exp[1])
    #plt.axis('equal') # Image non déformée
    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
    plt.title("Marche aléatoire d'une bactérie une probabilité de tourner P = sqrtC" )
    plt.ylabel("Pas verticaux")
    plt.xlabel("Pas horizontaux")
    
    
    if exp[2]==0 : #Si on tourne : 
    #plt.plot(exp[0][0],exp[1][0])#, marker="o", color="red", label="début") #Trace un point pour l'origine de départ de la bactérie
        plt.plot(exp[0][-1],exp[1][-1], marker=".", color="violet", label="fin") #Trace un point pour l'origine de départ de la bactérie
#plt.legend()
    else : 
        plt.plot(exp[0][-1],exp[1][-1], marker=".", color="gray", label="fin")
    test.append(exp[2])  
plt.show()

print(test)
#%% Tracé pour une seule bactérie


exp = marche_bacterie_concentration(N)
    
    
    
    # plt.figure()
    #plt.grid()
    
plt.figure()  
plt.imshow(M,extent=[-l,l,-l,l])
plt.plot(exp[0],exp[1])
    #plt.axis('equal') # Image non déformée
    #plt.xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
plt.title("Marche aléatoire d'une bactérie une probabilité de tourner P = C" )
plt.xlabel("Pas verticaux")
plt.ylabel("Pas horizontaux")
plt.plot(exp[0][0],exp[1][0], marker="o", color="red", label="début") #Trace un point pour l'origine de départ de la bactérie
plt.plot(exp[0][-1],exp[1][-1], marker="o", color="violet", label="fin") #Trace un point pour l'origine de départ de la bactérie
plt.legend()
plt.show()
