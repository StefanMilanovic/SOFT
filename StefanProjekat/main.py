# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 21:44:44 2018

@author: Stefan Milanovic RA66/2014
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure

from vektori import *

print "Hello world!"


def obucavanje():
    print("Pocetak obucavanja")
    ucitajMnist = fetch_mldata('MNIST original')
    podaciMnist = ucitajMnist.data
    preciznost = []
    
    #print podaciMnist[1]
    
    #grupa random podataka za 
   
     # print ucitajMnist.data.shape[0]
     
    #print grupa
    #grupe za treninranje = noviPodaci
    noviPodaci = podaciMnist[np.random.choice(ucitajMnist.data.shape[0],1000) ]
    lab = ucitajMnist.target.astype('int')
    noviLab = lab[np.random.choice(ucitajMnist.data.shape[0],1000) ]
    print ("primer lab :  %d" %noviLab[2]) 
    kVrednost = range(1, 30, 2) #range([start], stop[, step])
    
    #range creates a list, so if you do range(1, 10000000) it creates a list in memory with 9999999 elements.
    #xrange is a sequence object that evaluates lazily.
   
    testPodaci = podaciMnist[ np.random.choice(ucitajMnist.data.shape[0], 200)] 
    testLab= lab[ np.random.choice(ucitajMnist.data.shape[0], 200)] 
  
    
    for k in range (1,29,2):
        rezultat =KNeighborsClassifier(n_neighbors=k)
        rezultat.fit(noviPodaci,noviLab)
        #povecavanje preciznosti
        cilj = rezultat.score(testPodaci, testLab)
        #print("k=%d, %.1f%%   " % (k, cilj * 100))
       
        preciznost.append(cilj)
        
    # append - adds a single element to the end of the list
    #print len(preciznost)
    pom = preciznost[0]
    najveceK =0
    #obucavanje koje k ima najvecu sansu da bude tacno
   
    for i in range(0, len(preciznost)):
        #print preciznost[i]
        
        if (preciznost[i] >pom) :
            pom = preciznost[i]
            najveceK = i
        
    print najveceK

    konacniPodaci = prilagodiSliku(noviPodaci);
    
    rezultat = KNeighborsClassifier(n_neighbors=kVrednost[najveceK])
    rezultat.fit(konacniPodaci, noviLab)
    
    return rezultat;
    
    

def prilagodiSliku(noviPodaci):
    jezgroD = np.ones((2,2),np.uint8)
    jezgroE= np.ones((1,1),np.uint8)
    brIt = 1
    for n in range(0,len(noviPodaci)):
        #privremeno za testiranje
        
    #for n in range(0,3):
        podatak = noviPodaci[n].reshape((28,28)).astype("uint8")
       # exposure.rescale_intensity - Return image after stretching or shrinking its intensity levels.

        podatak = exposure.rescale_intensity(podatak, out_range=(0, 255))
        ret, image_bin = cv2.threshold(podatak, 100, 255, cv2.THRESH_BINARY) 
        
        dilatacija = cv2.dilate(image_bin,jezgroD,brIt)
        erozija = cv2.erode(image_bin,jezgroE,brIt)
        
        noviPodaciKontura = kontureSlike(image_bin,noviPodaci,n)
    #print podatak\
   
    img = Image.fromarray(podatak)
  #  img.show()
    img = Image.fromarray(dilatacija)
  #  img.show()
    img = Image.fromarray(erozija)
    #img.show()
   # kao prvi parametar prima sliku koja se binarizuje,
   #kao drugi parametar prima prag binarizacije, 
   #treći parametar je vrednost rezultujućeg piksela ako je veći od praga (255=belo),
   #poslednji parametar je tip thresholda
    return noviPodaciKontura

def kontureSlike(slika,noviPodaci,n):
    vel = (28, 28)
    izmSlika = slika;
    inCubic = cv2.INTER_CUBIC
    slika123,konture,hi =cv2.findContours(slika,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
   # cv2.drawContours(slika, konture, -1, (255, 0, 0), 1)
  #  img = Image.fromarray(slika)
 #   img.show()
    for idx  in konture:
        x,y,sirina,visina = cv2.boundingRect(idx);
        isecenaSlika=izmSlika[y:y+visina,x: x+sirina]; #secemo sliku
        isecenaSlika =  cv2.resize(isecenaSlika, vel,inCubic  )
        
        isecenaSlika=isecenaSlika.flatten();#iz mat vektor
        isecenaSlika = np.reshape(isecenaSlika, (1,-1))
                   
       
        noviPodaci[n]=isecenaSlika; 
        
    return noviPodaci



def ucitajVideo():
    
    jezgro = np.ones((2,2),np.uint8)
    brojac=0 
    true = 1
    snimak = "video-2.avi"
    capture = cv2.VideoCapture(snimak)
    #uzima frejm  po frejm
    
    while(true) :
        
        povratna, frejm = capture.read()
     
        if povratna:
           # frejm=cv2.cvtColor(frejm,cv2.COLOR_BGR2RGB)

            pomFrejm = frejm
            brojac = 1 + brojac
          #  print "ima slika"
           # print brojac
           #vadim liniju
            lower_range = np.array([ 125,0, 0])
            upper_range = np.array([255,100, 100])
            pomFrejm=cv2.inRange(pomFrejm,lower_range,upper_range)
             
            
            pomFrejm = cv2.dilate(pomFrejm,jezgro)
                #Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
#                where the arguments are:
#                
#                detected_edges: Source image, grayscale
#                #detected_edges: Output of the detector (can be the same as the input)
#                lowThreshold: The value entered by the user moving the Trackbar
#                highThreshold: Set in the program as three times the lower threshold (following Canny’s recommendation)
#                kernel_size: We defined it to be 3 (the size of the Sobel kernel to be used internally)
           # pomFrejm = cv2.cvtColor(pomFrejm, cv2.COLOR_BGR2GRAY)
            pomFrejm = cv2.Canny(pomFrejm,50,150,apertureSize = 3)         
            minLineLength = 100
            maxLineGap =2
            
          #  print "HoughLinesP :"   
            linije =cv2.HoughLinesP (pomFrejm, 1, np.pi / 180, 40, minLineLength, maxLineGap)
            
            a,b,c = linije.shape
            for i in range(a):
                cv2.line(frejm, (linije[i][0][0], linije[i][0][1]), (linije[i][0][2], linije[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
           # if brojac == 500 :
              #  img = Image.fromarray(frejm)
                #img.show()
            
            
            minx = linije[len(linije)-3][0][0]#min
            miny = linije[len(linije)-3][0][1]
            maxx = linije[len(linije)-3][0][2]
            maxy = linije[len(linije)-3][0][3]#max
                                
           
            for i in  range(0,len(linije)):
                x1 = linije[i][0][0]
                y1 = linije[i][0][1]
                x2 = linije[i][0][2]
                y2 = linije[i][0][3]
                if y1 > 62 :
                    if  x1 < minx:
                        miny = y1
                        minx = x1
                    if  x2 > maxx:
                        maxx = x2
                        maxy = y2
           
            ivice = [(minx, miny), (maxx, maxy)]
            mojaLinija = np.polyfit([minx,maxx],[miny,maxy],1)
            
            
            
            kontura = pronadjiBroj(brojac, frejm)
            if brojac == 500 :
                pomFrejm2 = pomFrejm
                pomFrejm3 = frejm
        else:
            print "nema vise slika, broj slika:"
            print brojac
            
            break
    
    img = Image.fromarray(pomFrejm2)
    img.show()
    img = Image.fromarray(pomFrejm3)
    img.show()
    
    #kada smo izvukli linniju i njene ivice  formiramo jed.
    capture.release()
    cv2.destroyAllWindows()


def pronadjiBroj(brojac,frejm):
     jezgroD=np.ones((2,2))
     pocetnaSlika = frejm;
     jezgroE=np.ones((1,1));
     k =1
     ret, slika =  urediSliku(frejm)
    
    # img = Image.fromarray(slika)
    # img.show()
     isecSlika=slika#imam slikku izdvojenih brojeva i sada cemo uraditi konture
     slika=cv2.dilate(slika,jezgroD,k);
     
     slika=cv2.erode(slika,jezgroE,k);
  
    #contours je lista pronađeih kontura
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
     slika,konture,hie =cv2.findContours(slika,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
             
     
     
     img = slika.copy()
    
     kontureVrednost=[];
     
     for kk in konture:
          koordX,koordY,sirina,visina = cv2.boundingRect(kk); 
          isecSlika=isecSlika[koordY:koordY+visina,koordX :koordX+sirina];
          pocetnaSlika = cv2.rectangle(pocetnaSlika,(koordX,koordY),(koordX+sirina,koordY+visina),(0,0,255),2)
    
     if brojac == 500:# samo radi primera jedne slike
        print "Prikaz slike iscrtanih kontura"
        img = slika.copy()
        cv2.drawContours(img, konture, -1, (255, 0, 0), 1)
        img = Image.fromarray(img)
        
        img.show()
        


     return kontureVrednost;
    
    
def urediSliku(pomFrejm) :
  #  gray = cv2.imread(pomFrejm,0)
      
    siviFrejm = cv2.cvtColor(pomFrejm, cv2.COLOR_BGR2GRAY) 
    # ret je vrednost praga, image_bin je binarna slika
    ret,slika=cv2.threshold(siviFrejm,200, 255, cv2.THRESH_BINARY);
    return ret, slika;



print "pocetak obucavanja.."
rez = obucavanje()
print "kraj obucavanja.."
print"ucitavanje i obrada snimka.."
ucitajVideo()
print"kraj ucitavanja i obrade snimka.."



    
    
    
    
    
    
    
    
    
    
    
    
