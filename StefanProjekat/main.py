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
from vektori import distance
from vektori import pnt2line
from vektori import unit
from vektori import add
from vektori import dot
from vektori import scale
from vektori import vector
from vektori import length

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
        
        slika2 = image_bin
        dilatacija = cv2.dilate(image_bin,jezgroD,brIt)
        erozija = cv2.erode(dilatacija,jezgroE,brIt)
        
        noviPodaciKontura = kontureSlike(erozija,noviPodaci,n,slika2)
    #print podatak\
   
    #img = Image.fromarray(podatak)
  # #img = Image.fromarray(dilatacija)
  #  img.show()
    #img = Image.fromarray(erozija)
    #img.show()
   # kao prvi parametar prima sliku koja se binarizuje,
   #kao drugi parametar prima prag binarizacije, 
   #treći parametar je vrednost rezultujućeg piksela ako je veći od praga (255=belo),
   #poslednji parametar je tip thresholda
    return noviPodaciKontura

def kontureSlike(slika,noviPodaci,n,slika2):
    vel = (28, 28)

    inCubic = cv2.INTER_CUBIC
    slika123,konture,hi =cv2.findContours(slika,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
   # cv2.drawContours(slika, konture, -1, (255, 0, 0), 1)
  #  img = Image.fromarray(slika)
 #   img.show()
    for idx  in konture:
        x,y,sirina,visina = cv2.boundingRect(idx);
        isecenaSlika=slika2[y:y+visina,x: x+sirina]; #secemo sliku
        isecenaSlika =  cv2.resize(isecenaSlika, vel,inCubic  )
        
        isecenaSlika=isecenaSlika.flatten();#iz mat vektor
        isecenaSlika = np.reshape(isecenaSlika, (1,-1))
                   
       
        noviPodaci[n]=isecenaSlika; 
        
    return noviPodaci



def ucitajVideo(rez):
    zbir =0
    print "Postavio zbir na nula"
    jezgro = np.ones((2,2),np.uint8)
    brojac=0 
    true = 1
    brojevi = []
    oz = -1 # jedinstevna identifikacija brojeva sa frejma
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
            
           # a,b,c = linije.shape
           # for i in range(a):
                #cv2.line(frejm, (linije[i][0][0], linije[i][0][1]), (linije[i][0][2], linije[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
           # if brojac == 500 :
              #  img = Image.fromarray(frejm)
                #img.show()
            
            #trazimo za liniju pocetnu i krajnju tacku
            pocetakLinX = linije[0][0][0]#min
            pocetakLinY = linije[0][0][0]
            krajLinX = linije[0][0][0]
            krajLinY = linije[0][0][0]#max
                        

            for j in  range(0,len(linije)):
                x1 = linije[j][0][0]
                x2 = linije[j][0][2]
                y1 = linije[j][0][1]             
                y2 = linije[j][0][3]
          
                if  x1 < pocetakLinX:
                       
                        pocetakLinX = x1
                        pocetakLinY = y1
                if  x2 > krajLinX:
                        krajLinX = x2
                        krajLinY = y2
                if y1 < pocetakLinY :
                    if  x1 < pocetakLinX:
                        pocetakLinX = x1
                        pocetakLinY = y1
                if y2 > pocetakLinY :
                    if  x2 > krajLinX:
                        krajLinX = x2
                        krajLinY = y2
           
            ivice = [(pocetakLinX, pocetakLinY), (krajLinX, krajLinY)]
            z = np.polyfit([pocetakLinX,krajLinX],[pocetakLinY,krajLinY],1)
            
            mojaLinija =z,[pocetakLinX,krajLinX],[pocetakLinY,krajLinY]
            kontura = pronadjiBroj(brojac, frejm,oz)
           
            
            uzmiBroj(kontura,brojac,frejm,brojevi,oz) # najbitnije da postavlja vrednost broja 
           
            suma=  presekIZbir(brojevi,brojac,mojaLinija,zbir,rez)
            zbir = suma;
           
            #ispis za jednu sliku
            if brojac == 500 :
                pomFrejm2 = pomFrejm
                pomFrejm3 = frejm
                print "Minimumi x"
                print pocetakLinX
                print "Minimumi y"
                print pocetakLinY
                print "Maksimum x"
                print krajLinX
                print "Maksimum y"
                print krajLinY
                print "REZULTAT TRENUTNI frejm 500: :"
                print zbir
            
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


def pronadjiBroj(brojac,frejm,oz):
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
     slikaNova,konture,hie =cv2.findContours(slika,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
             
     kontureVrednost=[];
     oznaka =[]; # bice za svaki broj
     sabran = False
     for kk in konture:
          koordX,koordY,sirina,visina = cv2.boundingRect(kk); 
          isecSlika2=isecSlika[koordY:koordY+visina,koordX :koordX+sirina];
          pocetnaSlika = cv2.rectangle(pocetnaSlika,(koordX,koordY),(koordX+sirina,koordY+visina),(0,0,255),2)
          oznaka = [oz,(koordX+(sirina/2),koordY+(visina/2)), [visina,sirina],isecSlika2,sabran,brojac]
          #jedinstvena vr. koord sredine,velicina i slika
          
          kontureVrednost.append(oznaka)
          
     if brojac == 500:# samo radi primera jedne slike
        print "Prikaz slike iscrtanih kontura"
        img123 = slika.copy()
        cv2.drawContours(img123, konture, -1, (255, 0, 0), 1)
        img123 = Image.fromarray(img123)
        img123.show()
        img123 = Image.fromarray(isecSlika2)
        img123.show()

     return kontureVrednost;
    
    
def urediSliku(pomFrejm) :
  #  gray = cv2.imread(pomFrejm,0)
      
    siviFrejm = cv2.cvtColor(pomFrejm, cv2.COLOR_BGR2GRAY) 
    # ret je vrednost praga, image_bin je binarna slika
    ret,slika=cv2.threshold(siviFrejm,200, 255, cv2.THRESH_BINARY);
    return ret, slika;


def uzmiBroj(kontura,brojac,frejm,brojevi,oz):
    
    for cifra in kontura:
        brProsli = najblizi(cifra,brojevi)
        
        brZaObradu = len(brProsli)
        if brZaObradu ==0:
            oz = oz +1;
            cifra[0] = oz
            cifra[4] = False # sabran
            brojevi.append(cifra);
        #sta ako je vise od jednog ? 
        elif brZaObradu==1:#    oznaka = [oz,(koordX+(sirina/2,koordY+(visina/2))), [visina,sirina],isecSlika,sabran,brojac]
            brProsli[0][1] = cifra[1]
            brProsli[0][5] = brojac
    
def najblizi(cifra, brojevi):
    prolaziBr = []
    for  h  in brojevi:
        if(distance(h[1],cifra[1])<20):
            prolaziBr.append(h)
          
    if len(prolaziBr)>1 : 

        return najbliziDrugi(cifra,prolaziBr)
    else:

   
        return prolaziBr



def najbliziDrugi(cifra, prolaziBr):
    minRastojanje = distance(cifra[1],prolaziBr[0][1])
    trenutniNaj = prolaziBr[0]
    nizNajblizih = []       
    
    for h in prolaziBr:
        if(distance(h[1],cifra[1])<minRastojanje):
            
            minRastojanje = distance(h[1],cifra[1]) 
            trenutniNaj = h;
    nizNajblizih.append(trenutniNaj)
    return nizNajblizih
    
def presekIZbir(brojevi,brojac,mojaLinija,zbir,rez):
    
    
    for i in brojevi:
        #print "frejm -  br[5]"
        trajanjeBr = brojac - i[5]
        #print brojac
        #print i[5]
        if(trajanjeBr <5):
            razdaljinaBrLin = pnt2line(i[1],(mojaLinija[1][0],mojaLinija[1][1]),(mojaLinija[2][0],mojaLinija[2][1]))
            
            
            if( not i[4] and razdaljinaBrLin <10):
                i[4] = True
                predvidjenaVrednost = prepoznajBroj(i[3],rez)
                zbir = zbir +predvidjenaVrednost
                #zbir = zbir+1
                print "predvidjeno pa suma"
                print predvidjenaVrednost
                print zbir
    
    return zbir
    
def prepoznajBroj(isecak,rez):
    
    
    dim = (28, 28)
 
    obradjenaSlika = cv2.resize(isecak, dim, interpolation = cv2.INTER_CUBIC)

    obradjenaSlika=obradjenaSlika.flatten();
    obradjenaSlika = np.reshape(obradjenaSlika, (1,-1))
    pogadjaj=rez.predict(obradjenaSlika)[0];
    
   
    
    return pogadjaj;
   

print "pocetak obucavanja.."
rez = obucavanje()
print "kraj obucavanja.."
print"ucitavanje i obrada snimka.."
ucitajVideo(rez)
print"kraj ucitavanja i obrade snimka.."



print"test primeri.."
n=3
distance2=[[[0]*n]*n]*n
print distance2
print "  "
#distance2[0] = 1 #ceo red jedan postaje 1
#distance2[0][0] = 1 #prva matrica svakog reda postaje 1
#distance2[0][0][0] = 1 # pristupamo jednom broju uutar tih matrica
distance2[2][2][1] = 2
print distance2 
    

    
    
