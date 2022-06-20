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

# from numba import jit
############################################################################################
#%% Récupération du film de suivi avec prise en compte du seuil (et récupération du seuil)


# @jit(nopython=True)
def fonction_film (Chemin,*parametres,**kwargs) :
    
    
    """
    Chemin : path où se trouvent les vidéos
    Cette fonction créé un film de particules détectées, récupère le seuil au cours du temps 
    et récupère le dataframe des trajectoires totales
    
    """
    
    diameter,separation,zone_recherche,seuil,memory,ecart_max,seuil_I,film_dl=parametres
    
    #Chemins d'accès
    Video= "Pos0"
    # Video_suivi='Pos1'
    repertoire = 'sepa_'+str(separation)
    nom = "batch_"+str(repertoire)
    
    #Est-ce que le film existe déjà ?
    file_exists = os.path.exists(str(Chemin)+'//'+str(repertoire)+".pkl")
    
    if film_dl == 1 :
        pass
    elif file_exists == True:
        print('Le film existe déjà')
        return pd.read_pickle(str(Chemin)+'//'+str(repertoire)+".pkl")
    
    film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    
    plt.close()
    
    # Turn interactive plotting off
    import matplotlib
    matplotlib.use('Agg')
    #https://matplotlib.org/stable/users/explain/backends.html
    #Plus de détails ici
    
    f_batch=[]
    seuil_total =[]
    
    for i in tqdm(range(len(film))) :
        
        #Prise en compte du photo-blanchiment
        #On regarde la gaussienne sous le seuil
        f_test = tp.locate(film[i],diameter = diameter, invert=False, minmass = 1, separation=separation, noise_size=1,engine='auto')
        mean = np.mean(f_test.mass)
        std = np.std(f_test.mass)
        seuil= mean + 0.8*std
        
        #On détermine la moyenne de la nouvelle gaussienne et son std
        Gaussienne = []
    
        for row in f_test.mass :
            if row<seuil :
                Gaussienne.append(row)
              
        mean_g=np.mean(Gaussienne)
        std_g = np.std(Gaussienne)
        seuil_g = mean_g + std_g
        seuil_total.append(seuil_g)
        
        
        #On fait le film avec un minmass = seuil_g
        f_film = tp.locate(film[i],diameter = diameter, invert=False, minmass = seuil_g, separation=separation, noise_size=1,engine='auto')
        
        #On récupère l'image pour faire un seul fichier batch et faire link/filter plus tard 
        f_batch.append(f_film)
        
   
    if film_dl == 1 :
        
        for i in tqdm(range(len(film))) :
        
            a=tp.annotate(f_batch[i], film[i],plot_style={'markersize': diameter});
            plt.close()
        
            #On enregistre le film dans un répertoire qu'il faut créer au besoin
            if not os.path.exists(str(Chemin)+'//'+str(repertoire)):
                os.makedirs(str(Chemin)+'//'+str(repertoire))
            #On sauvegarde le fichier
            a.figure.savefig(str(Chemin)+'//'+str(repertoire)+'/img_'+str(i)+'.tif')
        
        
    matplotlib.use('Qt5Agg')
    #On remet le backend interactif 
    
    f_batch_f=pd.concat(f_batch, axis=0)
    
    f_batch_f.to_pickle(str(Chemin)+'//'+str(repertoire)+".pkl") #Sauvegarde du dataframe panda
    
    return f_batch_f#,seuil_total

#%%% Film filtré
# @jit(nopython=True)
def fonction_film_filtre (Chemin,Batch_film,*parametres,**kwargs) :
    
    """
    Chemin : path où se trouvent les vidéos
    Batch_film : Outpout[0] de fonction_film
    memoire : nombre de frame durant laquelle la particule peut disparaître et être retrouvée par le code ensuite
    seuil : Durée minimale d'une trajectoire [frames]
    zone_recherche : Déplacement maximal entre deux pertes de détection
    
    Cette fonction créé un film à partir des trajectoires filtrées (mémoire, seuil, etc)
    """
    
    diameter,separation,zone_recherche,seuil,memory,ecart_max,seuil_I,film_dl=parametres
    
    #Chemins d'accès
    Video= "Pos0"
    film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    repertoire = 'sepa_'+str(separation)+'_seuil_'+str(seuil)+'_mem_'+str(memory)
    nom = "batch_"+str(repertoire)
    
    #Est-ce que le film existe déjà ?
    file_exists = os.path.exists(str(Chemin)+'//'+str(repertoire)+".pkl")
    
    if film_dl == 1 :
        pass
    elif file_exists == True:
        print('Le film filtré existe déjà')
        return pd.read_pickle(str(Chemin)+'//'+str(repertoire)+".pkl")
    

    #On réindexe le film
    index=len(Batch_film)
    Batch_film_index=Batch_film.set_index(pd.Index(np.linspace(0,index-1,index).astype(int)))

    tp.quiet() #Pour que le programme n'affiche pas ses calculs
    Batch_film_link = tp.link(Batch_film_index, search_range= zone_recherche, memory=memory) #Lie les trajectoires
    Batch_film_filtre = tp.filter_stubs(Batch_film_link,threshold=seuil) #Filtre les trajectoires d'une longeur inférieures au threshold 
    

    # nbpart_frame=[]
    
    
    if film_dl == 1 :
        
        # Turn interactive plotting off
        import matplotlib
        matplotlib.use('Agg')
        
        #On enregistre le film dans un répertoire qu'il faut créer au besoin
        if not os.path.exists(str(Chemin)+'//'+str(repertoire)):
            os.makedirs(str(Chemin)+'//'+str(repertoire))
    
        for i in tqdm(range(len(film))) :
            a=tp.annotate(Batch_film_filtre.loc[i], film[i],plot_style={'markersize': diameter});
            # nbpart_frame.append(len(t_f1.loc[i])) #On récupère le nombre de particules à chaque frame
            plt.close()
    
            #On sauvegarde le fichier
            a.figure.savefig(str(Chemin)+'//'+str(repertoire)+'/img_'+str(i)+'.tif')
        
        
        matplotlib.use('Qt5Agg')
        #On remet le backend interactif 
    
    Batch_film_filtre.to_pickle(str(Chemin)+'//'+str(repertoire)+".pkl") #Sauvegarde du dataframe panda
    return Batch_film_filtre

#%%%

def fonction_fragmentation (Chemin,Batch_film_filtre,*parametres,**kwargs) :
    

     
      diameter,separation,zone_recherche,seuil,memory,ecart_max,seuil_I,film_dl=parametres
      
      #Film étudié
      Video= "Pos0"
      film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
      repertoire = 'frag_'+str(ecart_max)+'_sepa_'+str(separation)+'_seuil_'+str(seuil)+'_mem_'+str(memory)+'_Seuil_I_'+str(seuil_I)
      nom = "batch_"+str(repertoire)
      
      #Est-ce que le film existe déjà ?
      file_exists_frag = os.path.exists(str(Chemin)+'//'+str(repertoire)+".pkl")
      file_exists_mere = os.path.exists(str(Chemin)+'//'+str(repertoire)+"_mere"+".pkl")
      if film_dl == 1 :
          pass
      elif file_exists_frag == True and file_exists_mere == True :
          print('Le film filtré avec fragmentation existe déjà')
          return pd.read_pickle(str(Chemin)+'//'+str(repertoire)+".pkl"), pd.read_pickle(str(Chemin)+'//'+str(repertoire)+"_mere"+".pkl")
      
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
              
              if I+I_app < (1+seuil_I)* I_m and I+I_app > (1-seuil_I) * I_m :
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
         return pd.DataFrame({'frame' : stock}), pd.DataFrame({'frame' : stock})
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
      
      Batch_part_mere_concat.to_pickle(str(Chemin)+'//'+str(repertoire)+"_mere"+".pkl") #Sauvegarde du dataframe panda
      
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
      return Batch_Nouv_part_concat,Batch_part_mere_concat
#%%


def fonction_extraction_fragmentation (Chemin_film,nb_film,*parametres): 

    """
    nb_film : matrice (1,2) numéro du premier film et du dernier film
    """
    
     
        
    nb_frag = []
    nb_part = []
    nb_frame= []

    for i in range (1,nb_film+1) :
        
         
        num = "{:02d}".format(i)
        print('Film numéro '+str(num))
        Chemin = Chemin_film+str(num)   
    
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
                nb_part.append(Batch_film_filtre['particle'].drop_duplicates().index)#-Position_AIFIRA)
                nb_frame.append(Batch_film_filtre.frame.to_numpy())
                nb_frag.append(Batch_Nouv_part_concat['particle'].drop_duplicates().index-Position_AIFIRA) #Trouve la première position 
        
        
            #Récupération des intensités
            
        
        
        # Csurf=np.round(len(nb_frag)/len(film),1)
        # nb_fortuits=np.round(len(nb_part)/len(film),4)
    # print(nb_part,nb_frag)
    Csurf=[np.round(len(f)/len(film),1) for f in nb_frame]
    nb_fortuits_frame=[np.round(len(f)/len(film),4) for f in nb_frag]
    # nb_part_detected=[len(f) for f in nb_part] 
    
    return nb_part,nb_frame,nb_frag,Csurf,nb_fortuits_frame
    # return nb_part,nb_frag
    
#%%  Fonction AIFIRA
def fonction_AIFIRA (Chemin) :

    #AIFIRA
    AIFIRA='logfile'
    AIFIRA_exists = os.path.exists(str(Chemin) + '\\' +str(AIFIRA))
    if AIFIRA_exists == False: #Parfois il n'existe pas de logfile
        return []
            
    else :
            
  
        #Position_AIFIRA 
        data = np.genfromtxt(str(Chemin) + '\\' +str(AIFIRA),dtype='str',skip_header=0)
        Position_AIFIRA=np.where(data=='AIFIRA,')[0][0]+1
    
        return Position_AIFIRA   
    
    
#%%

def fonction_test(Chemin,*parametres,**kwargs) :
    
    
    """
    Chemin : path où se trouvent les vidéos
    Cette fonction créé un film de particules détectées, récupère le seuil au cours du temps 
    et récupère le dataframe des trajectoires totales
    
    """
    oui=kwargs
    #Chemins d'accès
    Video= "Pos0"
    file_exists = os.path.exists(str(Chemin)+'\\'+str(Video))
    # film = pims.ImageSequence(str(Chemin)+'\\'+str(Video)+'\\'+'img_*'+'.tif')
    if file_exists == True :
        return print('yes')
    else :
        
    # file_exists = os.path.exists('readme.txt')

        print(file_exists)
        print(kwargs)
        print(oui)
 

#%%% Moyenne glissante

def moyenne_glissante(x, w):
    """
    x : Données du problème
    w : Nombre d'éléments de la moyenne glissante
    mode : gère la taille de la matrice retounée, sielle possède ou non les effets de bords dus à la moy glissante (valid, full, same)
    """
    return np.convolve(x, np.ones(w), mode = 'same') / w


#%%

def affichage_hybride(parametre_normal,parametre_avec_default="valeur par défaut",*args,**kwargs):
    
   print(parametre_normal)
   print(parametre_avec_default)
   print(args)
   print(kwargs)
   
   return()

#%%

