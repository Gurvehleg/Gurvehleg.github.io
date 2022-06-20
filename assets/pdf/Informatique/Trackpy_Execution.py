#%% Cellule d'importation

import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

#Echelle de color en log
from matplotlib.colors import LogNorm

import pims as pims
import trackpy as tp #http://soft-matter.github.io/trackpy/dev/api.html


import os 
from os.path import exists

from tqdm import tqdm #Barre de progression

#Pour fonctionner il faut que Fonction.py soit dans le bon working directory
from Trackpy_Fonction import fonction_film,fonction_film_filtre,fonction_fragmentation,fonction_extraction_fragmentation 
from Trackpy_Fonction import fonction_AIFIRA
from Trackpy_Fonction import affichage_hybride,fonction_test,moyenne_glissante

from Trackpy_Plot import plot_hist_Nbfrag_vs_temps

# from numba import jit
############################################################################################
#%% Paramètres de détection
"""
diameter : diametre estimé des particules [pixel]
zone_recherche : distance en pixel autour de laquelle une particule est recherchée entre deux frames [pixel] #Surout utile lorsque memory >0
minmass : luminosité minimale [UA]
separation : séparation minimale entre deux objets [pixel]
seuil : durée minimale des trajectoires filtrées [frame]
memory : nb de frame sur lesquelles on autorise la détection à perdre une particule [frame]
"""

diameter = 15
# minmass = 1
separation = 4
zone_recherche = 8 
seuil=15
memory = 0
ecart_max = 8 
seuil_I = 0.4
film_dl = 0
parametres = [diameter,separation,zone_recherche,seuil,memory,ecart_max,seuil_I ,film_dl]
############################################################################################

#%% Code manuel pour un seul film

#%%% On crée le film
num=3
# Chemin = "Y:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
# Chemin = "Y:\\_INSIDE_L\\20220509\\S2_1_eau_Beta\\Champ_fixe_gain_max_led100\\PhotoB"+str(num)
Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(num)
# Chemin = "Y:\\_INSIDE_L\\20211130\\T4_TBE_863mus\\TBE_"+str(num)
Batch_film,seuil_total=fonction_film(Chemin,*parametres)

#On réindexe le film
index=len(Batch_film)
Batch_film_index=Batch_film.set_index(pd.Index(np.linspace(0,index-1,index).astype(int)))

#%%% On crée un film filtré

Batch_film_filtre=fonction_film_filtre(Chemin,Batch_film,*parametres)


#%% Création du film fragmentation
ecart_max=8 #Pixel
fonction_fragmentation (Chemin,Batch_film_filtre,*parametres) 

############################################################################################
#%% Analyse de toutes les données avec faisceau

nb_film = 40 #Nombre de film
nb_part_total = []
nb_frag_total = []
nb_part_detected=[]


Csurf_total=[]
nb_fortuits_frame_total=[]
nb_fortuits_total=[]
nb_part_detected_total=[]
# for i in range (1,nb_film+1) :
    
#     num = "{:02d}".format(i)
#     print('Film numéro '+str(num))
#     # Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_"+str(num)  #Faisceau fort
#     # Chemin = "Y:\\_INSIDE_L\\20201126\\710mus\\T4_"+str(num)  #Faisceau faible
#     Chemin = "Y:\\_INSIDE_L\\20211130\\T4_TBE_863mus\\TBE_"#+str(num)  #Faisceau faible
    
#     Csurf,nb_fortuits=fonction_extraction_fragmentation (Chemin,nb_film,*parametres)
Chemin1 = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_"#+str(num)  #Faisceau fort
Chemin2 = "Y:\\_INSIDE_L\\20211130\\T4_TBE_8630mus\\TBE_"#+str(num)  #Faisceau fort
Chemin3 = "Y:\\_INSIDE_L\\20211130\\T4_TBE_863mus\\TBE_"#+str(num)  #Faisceau faible
Chemin4 = "Y:\\_INSIDE_L\\20211130\\T4_TBE_4315mus\\TBE_"#+str(num)  #Faisceau faible
List_chemin = [Chemin1,Chemin2,Chemin3]#,Chemin4]    

for idx,Chemin in enumerate(List_chemin) :
    nb_part,nb_frame,nb_frag,Csurf,nb_fortuits_frame=fonction_extraction_fragmentation (Chemin,nb_film,*parametres)
    Csurf_total.append(Csurf)
    nb_fortuits_frame_total.append(nb_fortuits_frame)
    
    # nb_part_detected=[len(f) for f in nb_part] 
    # nb_part_detected_total.append(nb_part_detected)
    
    # nb_frag_total.append(nb_frag)
    if idx < 1:
        nb_frag_total.append([np.count_nonzero((np.array(f)>0) & (np.array(f)<245) for f in nb_frag)])
        nb_fortuits_total.append([np.round(len(f),4) for f in nb_frag])
        nb_part_detected.append([np.count_nonzero(np.array(f)<245) for f in nb_part])
        
    else :
        nb_frag_total.append([np.count_nonzero((np.array(f)>0) & (np.array(f)<230) for f in nb_frag)])
        nb_fortuits_total.append([np.round(len(f),4) for f in nb_frag])
        nb_part_detected.append([np.count_nonzero(np.array(f)<230) for f in nb_part])
    
    # # #Film étudié
    # Video= "Pos0"
    # film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    
# Csurf=[np.round(len(f)/len(film),1) for f in nb_part_total]
# nb_fortuits=[np.round(len(f)/len(film),4) for f in nb_frag_total]
#%%

plt.figure()
plt.plot(Csurf_total[0],nb_part_detected[0],'.',label='35500mus')
# plt.plot(Csurf_total[1],nb_part_detected[1],'.',label='T4_TBE_8630mus')
plt.plot(Csurf_total[2],nb_part_detected[2],'.',label='T4_TBE_863mus')


plt.title("Nombre de particules détectées par film en fonction \n de la densité surfacique d'ADN \n pour 18,6s après faisceau")
plt.xlabel("Densité surfacique [Nombre de structures moyenne par frame]")
plt.ylabel("Nombre de particules détectées par film")
plt.legend()
#%% plot Nombre de fortuits par frame en fonction de la densité surfacique d'ADN


plt.figure()

plt.plot(Csurf_total[0],nb_fortuits_frame_total[0],'.',label='35500mus',color='red')  
# plt.plot(Csurf_total[1],nb_fortuits_frame_total[1],'.',label='T4_TBE_8630mus')    
plt.plot(Csurf_total[2],nb_fortuits_frame_total[2],'.',label='T4_TBE_863mus')  
# plt.plot(Csurf_total[3],nb_fortuits_frame_total[3],'.',label='T4_TBE_4315mus')  
plt.title("Nombre de fortuits par frame en fonction de la densité surfacique d'ADN")
plt.xlabel("Densité surfacique [Nombre de structures moyenne par frame]")
plt.ylabel("Nombre de fortuits par frame")
plt.legend()

#%% plot Nombre de fortuits par frame en fonction de la densité surfacique d'ADN en NOMBRE


plt.figure()
plt.plot(Csurf_total[0],nb_fortuits_total[0],'.',label='35500mus',color='red')  
plt.plot(Csurf_total[1],nb_fortuits_total[1],'.',label='T4_TBE_8630mus')    
plt.plot(Csurf_total[2],nb_fortuits_total[2],'.',label='T4_TBE_863mus',color='black')
# plt.plot(Csurf_total[3],nb_fortuits_total[3],'.',label='T4_TBE_4315mus')    
plt.title("Nombre de fortuits en fonction de la densité surfacique d'ADN")
plt.xlabel("Densité surfacique [Nombre de structures moyenne par frame]")
plt.ylabel("Nombre de fortuits")
plt.legend()

#%% Moyenne glissante


C_tri_1, nb_fortuits_tri_1 = (list(t) for t in zip(*sorted(zip(Csurf_total[0],nb_fortuits_total[0]))))
C_tri_2, nb_fortuits_tri_2 = (list(t) for t in zip(*sorted(zip(Csurf_total[1],nb_fortuits_total[1]))))
C_tri_3, nb_fortuits_tri_3 = (list(t) for t in zip(*sorted(zip(Csurf_total[2],nb_fortuits_total[2]))))

m_glissante = 3



plt.plot(C_tri_1,nb_fortuits_tri_1,'.')
plt.plot(C_tri_1,moyenne_glissante(nb_fortuits_tri_1,m_glissante))

# plt.plot(C_tri_2,nb_fortuits_tri_2,'.')
# plt.plot(C_tri_2,moyenne_glissante(nb_fortuits_tri_2,m_glissante))

plt.plot(C_tri_3,nb_fortuits_tri_3,'.')
plt.plot(C_tri_3,moyenne_glissante(nb_fortuits_tri_3,m_glissante))

plt.xlim(10,70)


#%% Moyenne glissante par paquet

# ~~~~~~~~~~~~~Obsolète~~~~~~~~~~~~~ #

mean_1 = []
mean_2 = []
mean_3 = []

std_1=[]
std_2=[]
std_3=[]

for i in range(0,len(nb_fortuits_tri_1),5) :
    mean_1.append(np.mean(nb_fortuits_tri_1[i:i+4]))
    mean_2.append(np.mean(nb_fortuits_tri_2[i:i+4]))
    mean_3.append(np.mean(nb_fortuits_tri_3[i:i+4]))
    
    
    std_1.append(np.std(nb_fortuits_tri_1[i:i+4]))
    std_2.append(np.std(nb_fortuits_tri_2[i:i+4]))
    std_3.append(np.std(nb_fortuits_tri_3[i:i+4]))
    
std_1_norm=std_1/np.sqrt(len(std_1)-1)
std_2_norm=std_2/np.sqrt(len(std_2)-1)
std_3_norm=std_3/np.sqrt(len(std_3)-1)

plt.errorbar(mean_1, y, yerr=std_1_norm)

#%% Tri des concentrations surfaciques
C_total_1=[]

C_list=[]
C=10
compteur = 0
while compteur < 40 :
    
    if C_tri_1[compteur] < C :
        C_list.append(C_tri_1[compteur])
        compteur+=1
    else :
        C+=5
        if C_list == [] :
            pass
        else :
            C_total_1.append(C_list)
            C_list=[]
    if compteur == 40 :
        C_total_1.append(C_list)
        
        
C_total_2=[]  
C_list=[]  
C=10
compteur = 0  
while compteur < 40 :
    
    if C_tri_2[compteur] < C :
        C_list.append(C_tri_2[compteur])
        compteur+=1
    else :
        C+=5
        if C_list == [] :
            pass
        else :
            C_total_2.append(C_list)
            C_list=[]
    if compteur == 40 :
        C_total_2.append(C_list)
        
C_total_3=[]  
C_list=[]  
C=10
compteur = 0  
while compteur < 40 :
    
    if C_tri_3[compteur] < C :
        C_list.append(C_tri_3[compteur])
        compteur+=1
    else :
        C+=5
        if C_list == [] :
            pass
        else :
            C_total_3.append(C_list)
            C_list=[]
    if compteur == 40 :
        C_total_3.append(C_list)
    # print(C)
    # print(i)

mean_1 = []
mean_2 = []
mean_3 = []

std_norm_1=[]
std_norm_2=[]
std_norm_3=[]

Taille_1 = []
Taille_2 = []
Taille_3 = []

for c in C_total_1 :
    mean_1.append(np.mean(c))
    std_norm_1.append(np.std(c)/np.sqrt(len(c)-1))
    Taille_1.append(len(c))
    
for c in C_total_2 :
    mean_2.append(np.mean(c))
    std_norm_2.append(np.std(c)/np.sqrt(len(c)-1))
    Taille_2.append(len(c))
    
for c in C_total_3 :
    mean_3.append(np.mean(c))
    std_norm_3.append(np.std(c)/np.sqrt(len(c)-1))
    Taille_3.append(len(c))
    
count_1 = []
count_2 = []
count_3 = []

compteur=0
for i in Taille_1 :
    count_1.append(np.mean(nb_fortuits_tri_1[compteur:compteur+i]))
    compteur+=i
compteur=0
for i in Taille_2 :
    count_2.append(np.mean(nb_fortuits_tri_2[compteur:compteur+i]))
    compteur+=i
compteur=0
for i in Taille_3 :
    count_3.append(np.mean(nb_fortuits_tri_3[compteur:compteur+i]))
    compteur+=i
    
plt.errorbar(mean_1, count_1 , yerr=std_norm_1,fmt='.',label='500 protons/\u03BCm$^2$',color='r')
# plt.errorbar(mean_2, count_2 , yerr=std_norm_2,fmt='.',label='100p',color='m')
plt.errorbar(mean_3, count_3 , yerr=std_norm_3,fmt='.',label='10 protons/\u03BCm$^2$')
plt.xlabel('Concentration surfacique moyenne')
plt.ylabel("Nombre d'événèments détectés par le code")
plt.title("Nombre d'événèments détectés par le code en fonction \n de la concentration surfacique moyenne pour différentes fluences")
plt.xlim(25,50)
plt.legend()
#%% Metadata
metadata = 'metadata'
num=4
Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(num)  #Faisceau fort
# a=np.genfromtxt(Chemin + '\\'+ 'Pos0' + '\\'+str(metadata) +'.txt',dtype='str',delimiter=',',skip_header=0)

file = pd.read_csv(Chemin + '\\'+ 'Pos0' + '\\'+str(metadata) +'.txt')
#%%

with open(Chemin + '\\'+ 'Pos0' + '\\'+str(metadata) +'.txt') as f:
    # print(f.read())
    # print(f.read().find("SliceIndex"))
    # print(f.read()[6265])
    # np.where(f.read()=="SliceIndex")
    lines = f.readlines()
    # print(f.read().split('\n'))
    # if "SliceIndex" in f.read() :
    #     print(f.read().find("SliceIndex"))
    #     print('yata')
    # print(np.where(f.read()=="'SliceIndex': 0"))
import csv
with open(Chemin + '\\'+ 'Pos0' + '\\'+str(metadata) +'.txt') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for idx,row in enumerate(spamreader):
        if idx < 113290 :
            print(type(row))
            print(np.where(row=='"SliceIndex"'))
        'oui'
        # print(row)
        # print(', '.join(row))
#%% Analyse de toutes les données avec faisceau
nb_frag = []
nb_frame = []
nb_part = []
nb_part_detected = []
nb_film = 40 #Nombre de film

I_mere_total=[]
I_fille_total=[]
I_total=[]
for i in range (1,nb_film+1) :
    num = "{:02d}".format(i)
    print('Film numéro '+str(num))
    Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_"+str(num)  #Faisceau fort
    # Chemin = "Y:\\_INSIDE_L\\20201126\\710mus\\T4_"+str(num)  #Faisceau faible
    # Chemin = "Y:\\_INSIDE_L\\20211130\\T4_TBE_863mus\\TBE_"+str(num)  #Faisceau faible
    
    
    #AIFIRA
    AIFIRA='logfile'
    AIFIRA_exists = os.path.exists(str(Chemin) + '\\' +str(AIFIRA))
    if AIFIRA_exists == False: #Parfois il n'existe pas de logfile
        nb_part.append([])
        nb_frag.append([]) 
        nb_frame.append([])
        
    else :
        
        #Film étudié
        Video= "Pos0"
        film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
        
        #Position_AIFIRA 
        data = np.genfromtxt(str(Chemin) + '\\' +str(AIFIRA),dtype='str',skip_header=0)
        Position_AIFIRA=np.where(data=='AIFIRA,')[0][0]+1
    
        #Dataframes
        Batch_film=fonction_film(Chemin,*parametres)
        Batch_film_filtre=fonction_film_filtre(Chemin,Batch_film,*parametres)
        Batch_Nouv_part_concat,Batch_part_mere_concat=fonction_fragmentation (Chemin,Batch_film_filtre,*parametres) 
        
        if len(Batch_Nouv_part_concat) == 0 : #Si on ne détecte pas de fragmentation
            nb_frag.append([])
            nb_frame.append(Batch_film_filtre.frame.to_numpy())
            nb_part.append(Batch_film_filtre['particle'].drop_duplicates().index)
        else :
            nb_frame.append(Batch_film_filtre.frame.to_numpy())
            nb_part.append(Batch_film_filtre['particle'].drop_duplicates().index)
            nb_frag.append(Batch_Nouv_part_concat['particle'].drop_duplicates().index-Position_AIFIRA) #Trouve la première position 
    I=[]
    I_mere=[]
    I_fille=[]
    
    
    for i in pd.unique(Batch_film_filtre['particle'].drop_duplicates()) : #Stockage de l'intensité 
        I.append(np.mean(Batch_film_filtre.set_index('particle').loc[i].mass))
    I_total.append(I)    
        
    if len(Batch_Nouv_part_concat) == 0 : #Si on ne détecte pas de fragmentation  
        I_mere.append(I_mere) 
        I_fille.append(I_fille) 
        
    else :        
        for part in pd.unique(Batch_part_mere_concat.particle) : #récupère indice
            I_mere.append(np.mean(Batch_part_mere_concat.loc[Batch_part_mere_concat['particle'] == part].mass)) #récupère la moyenne des intesités
        for part in pd.unique(Batch_Nouv_part_concat.particle) : #récupère indice
            I_fille.append(np.mean(Batch_Nouv_part_concat.loc[Batch_Nouv_part_concat['particle'] == part].mass)) #récupère la moyenne des intesités
        I_mere_total.append(I_mere)
        I_fille_total.append(I_fille)
        
    
nb_part_detected=[len(f) for f in nb_part] 
#%%
I=[]
for i in pd.unique(Batch_film_filtre['particle'].drop_duplicates()) :
    I.append(np.mean(Batch_film_filtre.set_index('particle').loc[i].mass))
#%%   

#np.min(np.concatenate(I_mere_total))

plt.figure()
plt.hist(np.concatenate(I_mere_total),50)
# plt.hist(np.concatenate(I_total),50)
# plt.hist(np.concatenate(I_fille_total),20)
#%%         
    Csurf=[np.round(len(f)/len(film),1) for f in nb_part]
    nb_fortuits=[np.round(len(f)/len(film),4) for f in nb_frag]
    
    plt.plot(Csurf,nb_fortuits,'.')
#%%
#Plot
plot_hist_Nbfrag_vs_temps (Chemin,nb_frag,nb_part,*parametres) 




#%%

Population = []

nb_film = 10
for i in range (1,nb_film+1) :
    num = "{:02d}".format(i)
    print('Film numéro '+str(num))
    Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_"+str(num)  #Faisceau fort
    # Chemin = "Y:\\_INSIDE_L\\20201126\\710mus\\T4_"+str(num)  #Faisceau faible
    # Chemin = "Y:\\_INSIDE_L\\20211130\\T4_TBE_863mus\\TBE_"+str(num)  #Faisceau faible
    
    #Film étudié
    Video= "Pos0"
    film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    
    Position_AIFIRA=fonction_AIFIRA(Chemin)
    
    tp.quiet() 
    Batch = tp.batch(film[Position_AIFIRA:Position_AIFIRA+245], diameter = 15, minmass=7200, invert=False,separation=4, processes=1);
    Link = tp.link(Batch, 5, memory=0)
    Batch_filtre=tp.filter_stubs(Link,threshold=1)
    
    Population.append(len(Batch_filtre.loc[Position_AIFIRA]))
    
#%%

Population_totale = []
#%%
#%%
#%%
#%%
#%%
#%% Analyse de toutes les données sans faisceau TBE + beta



nb_frag = []
nb_part = []
nb_film = 18 #Nombre de film


for num in range (1,nb_film+1) :

    print('Film numéro '+str(num))
    Chemin = "Y:\\_INSIDE_L\\20220509\\S2_1_TBE_Beta\\Champ_fixe_gain_max_led100\\PhotoB"+str(num)
       
    Batch_film=fonction_film(Chemin,*parametres)
    Batch_film_filtre=fonction_film_filtre(Chemin,Batch_film,*parametres)
    Batch_Nouv_part_concat=fonction_fragmentation (Chemin,Batch_film_filtre,ecart_max,*parametres) 
    
    nb_part.append(Batch_film_filtre.frame.to_numpy())
    nb_frag.append(Batch_Nouv_part_concat.frame.to_numpy())
#%%
#Plot
plot_hist_Nbfrag_vs_temps (Chemin,nb_frag,nb_part,*parametres) 

#%%  Test fonction clear
Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_02"  #Faisceau faible

from os import listdir
from os.path import isfile, join, isdir
onlyfiles = [f for f in listdir(Chemin) if isdir(join(Chemin, f))]

for elem in onlyfiles :
    if elem == 'Pos0' :
        pass
    else :
        os.remove(join(Chemin, elem))

#%% 

Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(1)  #Faisceau fort
for z in range (0,1):
    

      
      diameter,separation,zone_recherche,seuil,memory,ecart_max,film_dl=parametres
      
      #Film étudié
      Video= "Pos0"
      film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
      repertoire = 'frag_'+str(ecart_max)+'_sepa_'+str(separation)+'_seuil_'+str(seuil)+'_mem_'+str(memory)
      nom = "batch_"+str(repertoire)
      
      #Est-ce que le film existe déjà ?
      file_exists = os.path.exists(str(Chemin)+'//'+str(repertoire)+".pkl")
      if file_exists == True:
          print('Le film filtré avec fragmentation existe déjà')
          # return pd.read_pickle(str(Chemin)+'//'+str(repertoire)+".pkl")
      
      #Détection des indices des nouvelles particules
      frame_max=max(Batch_film_filtre.frame)
      Batch_film_filtre_set_particule=Batch_film_filtre.set_index('particle')

      Nouv_part=[]

      for frame in range(frame_max):
          temp2=Batch_film_filtre.loc[frame+1].particle #Fichier temporaire de comparaison
          temp1=Batch_film_filtre.loc[frame].particle
          difference = list(set(temp2) - set(temp1)) #On fait une différence assymétrique entre les deux fichiers
          # temp3 = [x for x in temp1 if x not in s] 
          
          Nouv_part.append(difference) #Frame 0 n'existe pas
          
      ######### Stockage des données des nouvelles particules   #########
      
      Nouv_part_data = [] #frame, particule, x, y, intensite

      """
      A comparer en rapidité avec un array de longeur non zero Nouv_part
      """

      for frame in range(frame_max):
          if Nouv_part[frame] == []:
              pass
          else :
              for part in Nouv_part[frame] : #On regarde les particules créés à chaque frame
                  extraction= Batch_film_filtre.loc[Batch_film_filtre['particle'] == part].loc[frame+1] #Particule part à la frame frame+1
                  x=extraction.x
                  y=extraction.y
                  Int=extraction.mass
                  Nouv_part_data.append([frame+1,part,x,y,Int])
                  
      ######### Comparaison avec le fichier mère #########

      """
      Il est possible de faire directement la comparaison sur le dataframe

      https://datatofish.com/select-rows-pandas-dataframe/

      df.loc[(df[‘Color’] == ‘Green’) & (df[‘Shape’] == ‘Rectangle’)]
      """
      stock=[]
      part_mere=[]
      
      for elem in Nouv_part_data :
          (frame_app, part_app, x_app, y_app, I_app)=elem #apparition
          
          x= (Batch_film_filtre.loc[frame_app-1].x ).to_numpy() #On regarde les particules à la frame d'avant apparition et on le transforme en array
          y=(Batch_film_filtre.loc[frame_app-1].y).to_numpy()
          part =(Batch_film_filtre.loc[frame_app-1].particle).to_numpy() 
          # frame =(Batch_film_filtre.loc[frame_app-1].frame).to_numpy()
          
          # frame_2 =(Batch_film_filtre.loc[frame_app+1].frame).to_numpy() 
          part_3 =(Batch_film_filtre.loc[frame_app+3].particle).to_numpy() #particules à la frame d'apparition + 3
          
          
          distance = np.sqrt((x-x_app)**2+(y-y_app)**2)
          
          ide = np.nonzero(distance< ecart_max)[0] #indice de la particule mère qui est à une distance <r
          index=part[ide] #Indice particulaire de la particule mère
          if ide.size== 0 :  #Ne garde que les traj< ecart_max
               pass
          elif np.nonzero(np.where(part_3==index)[0])[0].size>0 and np.nonzero(np.where(part_3==part_app)[0])[0].size>0 : #Il faut que le fragment originel et la particule apparue existent 3 frames plus tard
              
              I_m = Batch_film_filtre.loc[Batch_film_filtre['particle'] == index[0]].loc[frame_app-1].mass #Intensité de la particule mère
              I=Batch_film_filtre.loc[Batch_film_filtre['particle'] == index[0]].loc[frame_app].mass#Intensité de l'autre fragment (mère)
              
              if I+I_app < 1.2* I_m and I+I_app > 0.8 * I_m :
                  part_mere.append(part[ide])
                  stock.append(elem)
              
          
          # if ide.size== 0 :  #Ne garde que les traj< ecart_max
          #     pass
          # else :
          #     for i_m in I_m : #Il peut y avoir plusieurs particules qui respectent le critère de distance
          #         for i in I :
          #             if I_app+i<1.1*i_m and I_app+i>0.9*i_m and np.nonzero(np.where(part_1==part[ide])[0])[0].size>0 : #Condition sur l'intensité et il faut que la particule mère existe 3 frames plus tard
          #                 part_mere.append(part[ide])
          #                 stock.append(elem)
         
      if stock == [] :
         print("Pas d'évenement de fragementation détecté")
         # return pd.DataFrame({'frame' : stock}) 
      # Batch concaténé des particules qui viennent d'apparaitre
      Batch_Nouv_part=[]
     
      for part in np.array(stock,dtype=int)[:,1] : #indice des particules
          Batch_Nouv_part.append(Batch_film_filtre.loc[Batch_film_filtre['particle'] == part])
             
      Batch_Nouv_part_concat=pd.concat(Batch_Nouv_part) 
      
      # Batch concaténé des particules mère
      Batch_part_mere=[]
      for part in part_mere : #indice des particules
          Batch_part_mere.append(Batch_film_filtre.loc[Batch_film_filtre['particle'] == part[0]])
          
      Batch_part_mere_concat=pd.concat(Batch_part_mere)  
      
      
      if film_dl == 1 :
      
          #Titre
          AIFIRA='logfile'
          AIFIRA_exists = os.path.exists(str(Chemin) + '\\' +str(AIFIRA))
          if AIFIRA_exists == True:
              data = np.genfromtxt(str(Chemin) + '\\' +str(AIFIRA),dtype='str',skip_header=0)
              Position_AIFIRA=np.where(data=='AIFIRA,')
              Title = 'Faisceau à la frame '+str(Position_AIFIRA[0][0]+1)
                
          else :
              Title = 'Film sans faisceau'
            
              
              
          #Création du batch 
          # Turn interactive plotting off
          import matplotlib
          matplotlib.use('Agg')
    
          #On enregistre le film dans un répertoire qu'il faut créer au besoin
          if not os.path.exists(str(Chemin)+'//'+str(repertoire)):
              os.makedirs(str(Chemin)+'//'+str(repertoire))
    
    
          for frame in tqdm(range(len(film))) :
              plt.close()
              a=tp.annotate(Batch_film_filtre.loc[frame], film[frame],plot_style={'markersize': diameter});
              plt.title(Title)
              
              if frame in pd.unique(Batch_Nouv_part_concat.frame) :
                  
                  a=tp.annotate(Batch_Nouv_part_concat.loc[frame], film[frame],plot_style={'markersize': diameter}, color= "Blue")
                  
              
              if frame in pd.unique(Batch_part_mere_concat.frame) :
              
                  a=tp.annotate(Batch_part_mere_concat.loc[frame], film[frame],plot_style={'markersize': diameter}, color= "Green")
                  
                  
                  
    
              #On sauvegarde le fichier
              a.figure.savefig(str(Chemin)+'//'+str(repertoire)+'/img_'+str(frame)+'.tif')
              
          matplotlib.use('Qt5Agg')
          #On remet le backend interactif 
      
      Batch_Nouv_part_concat.to_pickle(str(Chemin)+'//'+str(repertoire)+".pkl") #Sauvegarde du dataframe panda
      # return Batch_Nouv_part_concat

#%% 

#%% 
num=1
Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(num)
AIFIRA='logfile'
#Position_AIFIRA 
data = np.genfromtxt(str(Chemin) + '\\' +str(AIFIRA),dtype='str',skip_header=0)
Position_AIFIRA=np.where(data=='AIFIRA,')[0][0]+1
#%% 

frame_list = []
nb_part = []

for num in range (1,2) :
    
    Chemin = "Y:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(num)
    # frame_list.append(pd.unique(Batch_Nouv_part_concat.frame))
    for frame in pd.unique(Batch_Nouv_part_concat.frame) :
        frame_list.append(frame)
        nb_part.append(np.size(Batch_Nouv_part_concat.particle.loc[frame]))


############################################################################################
#%% Récupération du nombre de particules par frame pour chaque film avec faisceau

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Obsolète~~~~~~~~~~~~~~~~~~~~~~~~~~~#

AIFIRA='logfile'

nb_part_total =[]

for num in range (4,7) :
    
    print('film numéro ' +str(num) )
    Chemin = "G:\\_INSIDE_L\\20201127\\35500mus\\T4_0"+str(num)
    # film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    #Position_AIFIRA 
    data = np.genfromtxt(str(Chemin) + '\\' +str(AIFIRA),dtype='str',skip_header=0)
    Position_AIFIRA=np.where(data=='AIFIRA,')


    (Batch_film,seuil_total)=fonction_film(Chemin,*parametres)

    
    nb_part=fonction_film_filtre (Chemin,Batch_film,diameter,seuil,zone_recherche,memory)
    nb_part_total.append(nb_part)

#%%% plot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Obsolète~~~~~~~~~~~~~~~~~~~~~~~~~~~#
temps = np.linspace(1,666,num=666)*76/1000 #En général 76ms par frames

plt.figure()

for idx,i in enumerate(nb_part_total) :
    # print(len(i))
    plt.plot(temps,i,'.',label='Film numéro ' +str(idx))
    
plt.xlabel('Temps [s]')
plt.ylabel('Nombre de particule')
    # plt.plot(temps,moyenne_glissante(seuil_total,m_glissante),label='Moyenne glissante sur ' + str(m_glissante) + ' frames')
    # plt.axvline(x=Faisceau, color='r', label = "Faisceau de protons à t=" +str(round(Faisceau,1)) + 's',linestyle='--')
    # plt.ylim(min(seuil_total)*0.999,max(seuil_total)*1.001)
plt.title('Evolution du nombre de particule en fonction du temps pour plusieurs films \n mémoire de ' +str(memory) + "\n 500 protons/\u03BC$m^2$ & 35.5ms d'irradiation")
plt.legend()

############################################################################################

#%% Récupération des batch de chaque film sans irradiation



nombre_de_film = 3

Batch_film_total=[]

for num in range (1,nombre_de_film+1) :
    
    
    print('film numéro ' +str(num) )
    # Chemin = "G:\\_INSIDE_L\\20201002\\75ms_bin2_"+str(num)
    #Perso
    # Chemin = "G:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
    
    #Bureau
    # Chemin = "G:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
    Chemin = "Y:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
    # film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')

    #Création du batch
    (Batch_film,seuil_total)=fonction_film(Chemin,*parametres)
    #Sauvegarde des données de chaque film
    Batch_film_total.append(Batch_film)
    

#%%% Récupération des trajectoires filtrées



Seuil = [3] #C'est débile de faire pour des seuils plus grand que le seuil min...

traj_filtred_total =[] #Toutes les trajectoires de chaque film

for seuil in Seuil :

    traj_filtred_seuil=[] #Trajectoires pour un seuil donné
    
    for num in range (0,nombre_de_film) : #Nombre de film étudié
        print('film numéro ' +str(num+1) )
        # Chemin = "G:\\_INSIDE_L\\20201002\\75ms_bin2_"+str(num)
        #Perso
        # Chemin = "G:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
        
        #Bureau
        # Chemin = "G:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num)
        Chemin = "Y:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\photoB_600img_continues"+str(num+1)
        # film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
        
        Batch_film=Batch_film_total[num]
        (nbpart_frame,t_f1)=fonction_film_filtre(Chemin,Batch_film,*parametres)
    
        #On réindexe le film
        # index=len(f_batch_f_total[i])
        # f_batch_f_index=f_batch_f_total[i].set_index(pd.Index(np.linspace(0,index-1,index).astype(int)))
    
        # tp.quiet() #Pour que le programme n'affiche pas ses calculs
        # t_f = tp.link(f_batch_f_index, search_range= zone_recherche, memory=memory) #Lie les trajectoires
        # t_f1 = tp.filter_stubs(t_f,threshold=seuil) #Filtre les trajectoires d'une longeur inférieures au threshold
    
        traj_filtred_seuil.append(t_f1)
    
    traj_filtred_total.append(traj_filtred_seuil)
    
    

#%%% Plot pour chaque film

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Obsolète~~~~~~~~~~~~~~~~~~~~~~~~~~~#

for i in range(len(f_batch_f_total)) :
    #On réindexe le film
    index=len(f_batch_f_total[i])
    f_batch_f_index=f_batch_f_total[i].set_index(pd.Index(np.linspace(0,index-1,index).astype(int)))
    
    tp.quiet() #Pour que le programme n'affiche pas ses calculs
    t_f = tp.link(f_batch_f_index, search_range= zone_recherche, memory=memory) #Lie les trajectoires
    t_f1 = tp.filter_stubs(t_f,threshold=seuil) #Filtre les trajectoires d'une longeur inférieures au threshold
    
    plt.figure()
    plt.hist(t_f1.mass, bins=50,label='Film ')
    plt.yscale('log') 
    plt.xlabel('Intensité [UA]')
    plt.ylabel('Nombre de particule')
    plt.legend( )

#%%% subplot pour chaque seuil

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Obsolète~~~~~~~~~~~~~~~~~~~~~~~~~~~#

nb_colonne = int(len(Seuil)/2)


fig, axes = plt.subplots(nb_colonne,2)
fig.suptitle("Nombre de particule détectées en fonction de l'intensité pour un diametre de detection de d = " +str(diameter) +' pixels ' +"\n memory = " +str(memory) +" pi"
             +"; separation = " +str(separation) +" pi"+"; zone_recherche = " +str(zone_recherche)+" pi" + "\n sans irradiation")
fig.text(0.5, 0.04, 'Intensité [UA]', ha='center')
fig.text(0.08, 0.35, 'Nombre de particule détectées', ha='center', rotation='vertical')

for j in range(nb_colonne) :
    for i in range (0,2) :
        
        traj_filtred_total_concat=pd.concat(traj_filtred_total[j+i*nb_colonne],axis=0) #On concatène chaque dataframe de chaque seuil (j+i*nb_colonne couvre tout les entiers par incrément nb colonne)
        axes[j,i].set(yscale='log')
        axes[j,i].hist(traj_filtred_total_concat.mass, bins=50,label='Durée minimale de trajectoire = '+str(Seuil[j+i*nb_colonne])+' frames')
        axes[j,i].plot(0,0,label='Nb particule = '+str(len(traj_filtred_total_concat.mass)))
        axes[j,i].legend()
        # axes[j,i].grid(True,which="both", linestyle='--')
       

#%%% Histogramme 2D

# seuil=s+3
s=0

#3 traj filtrées 

num_film = len(traj_filtred_total[s])

I_moyenne_total =[] #Intensité moyenne des num_films
I_max_total=[] #Intensité maximale des num_films
N_frame_total = [] #Nombre de frame totale des num_films



for i in tqdm(range(num_film)) :
    
    #Extraction de l'indice des particules dans le dataframes des trajectoires filtrées
    particules = pd.unique(traj_filtred_total[s][i].particle)
    
    #Maintenant que les particules sont connues, on réindexe le film en particule et on peut extraire les trajectoires des particules
    traj_filtred_particule = traj_filtred_total[s][i].set_index('particle')
    
    
    I_moyenne = []  #Intensité moyenne de chaque trajectoires d'un film
    N_frame = np.zeros(len(particules))    #Durée de chaque trajectoires d'un film
    I_max =[]       #Intensité maximale de chaque trajectoires d'un film
    
    for idx,indice in enumerate(particules) :
        
        #On garde la trajectoire de chaque particule 'indice'
        trajectoire = traj_filtred_particule.loc[indice] 
        
        #On peut en extraire l'intensité moyenne 
        I_moyenne.append((trajectoire.mass).mean())
        #Et la durée en frame de la trajectoire
        N_frame[idx]=len(trajectoire)
        #Et l'instensité max
        I_max.append((trajectoire.mass).max())
        
    # Une fois que le film est exploré, on stocke les données pour passer au film suivant
    I_moyenne_total.append(I_moyenne)
    N_frame_total.append(N_frame)
    I_max_total.append(I_max)
    
#On récupère les données des n_films qu'on convertit en array pour les extraire en une ligne
N_frame_total_concat=np.concatenate(N_frame_total)
#%% Sauvegarde des données 

#Perso
# Chemin_sauvegarde = "G:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\Longueur_des_trajectoires"
#Bureau
Chemin_sauvegarde = "Y:\\_INSIDE_L\\20220422\\stackz_pas1mum_70ms\\Gain_max_led100\\Longueur_des_trajectoires"

np.savetxt(str(Chemin_sauvegarde)+'.txt',N_frame_total_concat)
#%%% Plot Histogramme 2D

#On récupère les données qu'on convertit en array pour les extraire en une ligne
x=np.array(N_frame_total,dtype=object).flatten()[0]
# y=np.array(I_moyenne_total,dtype=object).flatten()[0]
y=np.array(I_max_total,dtype=object).flatten()[0]


#Bining de l'histogramme
x_min = np.min(x)
x_max = np.max(x)

y_min = np.min(y)
y_max = np.max(y)
 
x_bins = np.linspace(x_min,x_max,x_max+1)
y_bins = np.linspace(y_min,y_max,100)


#Figure log log.
# plt.close('all')
plt.figure()
plt.xlabel('Nombre de frame sur laquelle une particule est suivie [Frame]')
plt.ylabel('Intensité maximale [UA]')
plt.xscale('log')
plt.yscale('log') 
plt.grid(True,which="both", linestyle='--')
plt.title("Intensité max d'une trajectoire en fonction de sa durée en frame")
plt.plot(x,y,'.')


#Histogramme 2D
plt.figure()
plt.xlabel('Nombre de frame sur laquelle une particule est suivie [Frame]')
plt.title("Intensité maximale d'une trajectoire en fonction de sa durée en frame")
plt.ylabel('Intensité maximale [UA]')
plt.xscale('log')
plt.yscale('log')
plt.hist2d(x,y,bins=[x_bins,y_bins],norm=LogNorm())
plt.colorbar(label='Nombre de trajectoire')

#%%% Histogramme

Seuil_exp=15

#On récupère les données des n_films qu'on convertit en array pour les extraire en une ligne
N_frame_total_concat=np.concatenate(N_frame_total)


sup_seuil = np.nonzero(N_frame_total_concat>Seuil_exp)[0] #Repère les indices des trajectoires supérieures au seuil
n_traj=len(sup_seuil) #On compte le nombre de trajectoires supérieures au seuil
x=N_frame_total_concat[sup_seuil].astype(int)
x_min,x_max = np.min(x),np.max(x)



plt.figure()
plt.hist(x,bins=range(x_min,x_max+1,1),label='Nombre de trajectoires = '+str(len(x)))
plt.yscale('log')
plt.xlim(0,130)
plt.xlabel('Nombre de frame sur laquelle une particule est suivie [Frame]')
plt.title("Nombre de frame sur laquelle une particule est suivie \n pour un seuil de "+str(Seuil_exp))
plt.ylabel('Nombre de trajectoire')
# plt.grid(True,which="both", linestyle='--')
plt.legend()
#%%% Projection de l'axe des ordonnées sur l'axe des abscisses

#On s'assure que notre liste de travail est un array
x=np.array(x)
#On repère chaque élément unique de l'axe des abscisses (durées de trajectoires)
frames = np.unique(x)
#On identifie le nombre de trajectoires qui on cette durée 
nb_traj=np.zeros(len(frames),dtype=int) #On créé un array qui va recevoir cette information

for idx,element in enumerate(frames) : #Va chercher chaque élément dans l'axe des abscisses 
    nb_traj[idx]=len(np.where(x==element)[0]) #On compte combien il y a d'élément sur l'axes des ordonnées sur chaque élément de l'axe des abscisses
#%%% Plot

plt.figure()
plt.title("Nombre d'élement détecté en fonction du nombre de frame \n sur lequel on fait le suivi")
plt.xlabel('Nombre de frame sur laquelle une particule est suivie [Frame]')
plt.ylabel('Nombre de particule suivies')
plt.grid(True,which="both", linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.plot(frames,nb_traj,'.')




#%%