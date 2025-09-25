from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from math import *
image=Image.open(r"C:\Users\Dell\Documents\Computer_Vision\image_fleurs.png")

image=image.resize((200,200))


#ici c_R est l'histogramme des valeur de rouge entre 0 et 255
#tandis ce que R est le canal brut,cad la valeur du pixel a un emplacement donné

def cdf(v,hist):
    res=0
    for i in range(0,v+1):
        res+=hist[i]
    return res

def h(v,M,N,hist):
    tot_pix=M*N
    cdf_v=cdf(v,hist)
    val=  int(((cdf_v-1)/(tot_pix-1))*255)
    return val

#200x200
M=200
N=200
box=20

#récupérations des cannaux RGB
list=np.asarray(image)
R=np.zeros((M,N),dtype="uint8")
G=np.zeros((M,N),dtype="uint8")
B=np.zeros((M,N),dtype="uint8")


for x in range(M):
    for y in range(N):
        R[x,y]=list[x,y,0]
        G[x,y]=list[x,y,1]
        B[x,y]=list[x,y,2]
        
    print(f"#1 {x}")


new_R=np.zeros((M,N),dtype="uint8")
new_G=np.zeros((M,N),dtype="uint8")
new_B=np.zeros((M,N),dtype="uint8")
for x in range(M):
    for y in range(N):
        
        if (x>=box/2 and y>=box/2) and (x<=M-(box/2) and y<=M-(box/2) ):
            image_crop=image.crop((x-(box/2),y-(box/2),x+(box/2),y+(box/2)))
            
        else:
            #dans le cas où nous somme sur un pixel dont tous les voisins ne sont pas definit on calcul la taille de ce que l'on a
            l,u,r,lo=x-(box/2),y-(box/2),x+(box/2),y+(box/2)
            if l<0:
                l=0
            elif u<0:
                u=0
            elif r>M:
                r=M
            elif lo>N:
                lo=N
            image_crop=image.crop((l,u,r,lo))
        
        #récupération des histogrammes
        count=image_crop.histogram()
        c_R=count[0:256]
        c_G=count[256:512]
        c_B=count[512:768]
        new_R[x,y]=h(R[x,y],box,box,c_R)
        new_G[x,y]=h(G[x,y],box,box,c_G)
        new_B[x,y]=h(B[x,y],box,box,c_B)
        
    print(f"#2 {x}")

new_array=np.zeros((M,N,3),dtype="uint8")

#chargement des valeurs vers la nouvelle image
for x in range(M):
    for y in range(N):
        new_array[x,y,0]=new_R[x,y]
        new_array[x,y,1]=new_G[x,y]
        new_array[x,y,2]=new_B[x,y]
    print(f"#3 {x}")
        
print(new_array)

#reconstitution de l'image
new_image=Image.fromarray(new_array,mode='RGB')
new_image.show()
image.show()
list=np.asarray(image)
new_list=np.asarray(new_image)
#ici liste vide pour l'affichage des histogrammes, en effet le plot à besoin d'une liste calssique et pas d'une matrice
R_s=[]
G_s=[]
B_s=[]
new_R_s=[]
new_G_s=[]
new_B_s=[]
for i in range(M):
    for j in range(N):
        R_s.append(list[i,j,0])
        G_s.append(list[i,j,1])
        B_s.append(list[i,j,2])
        new_R_s.append(new_list[i,j,0])
        new_G_s.append(new_list[i,j,0])
        new_B_s.append(new_list[i,j,0])

a,b=plt.subplots(3,2)
b[0,0].hist(R_s,bins=256,facecolor="red")
b[1,0].hist(G_s,bins=256,facecolor="green")
b[2,0].hist(B_s,bins=256,facecolor="blue")

b[0,1].hist(new_R_s,bins=256,facecolor="red")
b[1,1].hist(new_G_s,bins=256,facecolor="green")
b[2,1].hist(new_B_s,bins=256,facecolor="blue")

plt.show()
            
