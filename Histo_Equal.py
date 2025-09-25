from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
image=Image.open(r"C:\Users\Dell\Documents\Computer_Vision\image_fleurs.png")
#1627x1618
M=148
N=199
#récupérations des cannaux RGB
list=np.asarray(image)
R=np.zeros((M,N),dtype="uint8")
G=np.zeros((M,N),dtype="uint8")
B=np.zeros((M,N),dtype="uint8")
#ici liste vide pour l'affichage des histogrammes, en effet le plot à besoin d'une liste calssique et pas d'une matrice
R_s=[]
G_s=[]
B_s=[]
for i in range(M):
    for j in range(N):
        R[i,j]=list[i,j,0]
        G[i,j]=list[i,j,1]
        B[i,j]=list[i,j,2]
        
        R_s.append(list[i,j,0])
        G_s.append(list[i,j,1])
        B_s.append(list[i,j,2])
    print(f"#1 {i}")

#récupération des histogrammes
count=image.histogram()
c_R=count[0:256]
c_G=count[256:512]
c_B=count[512:768]
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
    return int(((cdf_v-1)/(tot_pix-1))*255)

"""#test

im_test=np.array([[52, 55, 61,  59,  79,  61, 76, 61],
                  [62, 59, 55, 104,  94,  85, 59, 71],
                  [63, 65, 66, 113, 144, 104, 63, 72],
                  [64, 70, 70, 126, 154, 109, 71, 69],
                  [67, 73, 68, 106, 122,  88, 68, 68],
                  [68, 79, 60,  70,  77,  66, 58, 75],
                  [69, 85, 64,  58,  55,  61, 65, 83],
                  [70, 87, 69,  68,  65,  73, 78, 90]],dtype="uint8")

im_test_s=Image.fromarray(im_test,mode='L')
im_test_s.show()
histo=im_test_s.histogram()

zero=np.zeros((8,8),dtype='uint8')
for x in range(8):
    for y in range(8):
        zero[x,y]=h(im_test[x,y],8,8,histo)
print(zero)
im_test_s=Image.fromarray(zero,mode='L')
im_test_s.show()"""


new_R=np.zeros((M,N),dtype="uint8")
new_G=np.zeros((M,N),dtype="uint8")
new_B=np.zeros((M,N),dtype="uint8")
for x in range(M):
    for y in range(N):
        new_R[x,y]=h(R[x,y],M,N,c_R)
        new_G[x,y]=h(G[x,y],M,N,c_G)
        new_B[x,y]=h(B[x,y],M,N,c_B)
    print(f"#2 {x}")

new_im=np.zeros((M,N,3),dtype="uint8")

for x in range(M):
    for y in range(N):
        new_im[x,y,0]=new_R[x,y]
        new_im[x,y,1]=new_G[x,y]
        new_im[x,y,2]=new_B[x,y]
    print(f"#3 {x}")

print(new_im)



new_im=Image.fromarray(new_im,mode='RGB')
new_im.show()
image.show()

new_R_s=[]
new_G_s=[]
new_B_s=[]
for i in range(M):
    for j in range(N):
        new_R_s.append(new_R[i,j])
        new_G_s.append(new_G[i,j])
        new_B_s.append(new_B[i,j])
    print(f"#4 {i}")



a,b=plt.subplots(3,2)
b[0,0].hist(R_s,bins=256,facecolor="red")
b[1,0].hist(G_s,bins=256,facecolor="green")
b[2,0].hist(B_s,bins=256,facecolor="blue")

b[0,1].hist(new_R_s,bins=256,facecolor="red")
b[1,1].hist(new_G_s,bins=256,facecolor="green")
b[2,1].hist(new_B_s,bins=256,facecolor="blue")

plt.show()