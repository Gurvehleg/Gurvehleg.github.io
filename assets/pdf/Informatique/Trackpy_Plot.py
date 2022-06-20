#%% Cellule d'importation

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3


import matplotlib.pyplot as plt


import numpy as np
import pandas as pd


import pims as pims
import trackpy as tp #http://soft-matter.github.io/trackpy/dev/api.html



import os 
from os.path import exists

from tqdm import tqdm #Barre de progression
#%%

#%%


def plot_hist_Nbfrag_vs_temps(Chemin,nb_frag,nb_part,*parametres) : 
    
    
    Video= "Pos0"
    film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    
    nb_film = len(nb_frag) #Nombre de film
    diameter,separation,zone_recherche,seuil,memory,ecart_max,seuil_I,film_dl=parametres
    
    x_min=np.min(np.concatenate(nb_frag)*76/1000)
    x_max=np.max(np.concatenate(nb_frag)*76/1000)

    # x_bins = np.linspace(x_min,x_max,abs(x_max-x_min+1))


    plt.figure()
    plt.hist(np.concatenate(nb_frag)*76/1000,bins=int(x_max-x_min+1),label= 'Faisceau à t = 0 s & '+str(len(np.concatenate(nb_frag)))+' particules')
    plt.xlabel('Temps après faisceau [s]')
    plt.title("Nombre d'évènements détectées en fonction du temps après faisceau \n pour un écart max de "+str(ecart_max)  +" et " +str(nb_film) +" films")
    plt.ylabel("Nombre d'évènements")
    plt.legend()

    x_min=np.min(np.concatenate(nb_part)*76/1000)
    x_max=np.max(np.concatenate(nb_part)*76/1000)

    # surface = 501*502*0.250**2*1000/76
    Csurf = len(np.concatenate(nb_part))/len(nb_part)/len(film)
    plt.figure()
    # Sans_t0=np.concatenate(nb_part)[np.nonzero(np.concatenate(nb_part))] #On repère les indices des frames > 0
    # plt.hist(Sans_t0*76/1000,bins=int(x_max-x_min+1),label= 'Faisceau à t = 0 s')
    plt.hist(np.concatenate(nb_part),bins=int(x_max-x_min+1),label= 'Faisceau à t = 0 s')
    plt.xlabel('Temps [s]')
    plt.title("Nombre de particules détectées en fonction du temps après faisceau \n pour "+str(nb_film) +" films \n concentration surfacique de "
              +str(np.round(Csurf,1))+" brins d'ADN par unité de surface")
    plt.ylabel("Nombre de particules détectées")
    plt.legend()
#%%


#%%

#%%
#%%


#%%
