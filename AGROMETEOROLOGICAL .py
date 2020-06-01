
import pandas as pd
import numpy as np

evapo = pd.read_csv('evapo2.csv')

bandas = pd.read_csv('ordem.csv')
amostra = pd.read_csv('prod.csv')

cab = tuple (bandas.head(0))

v = bandas[cab[0]].tolist()

datas = []
ponto = 0

for i in v:
    
    cena = i + '.csv'
    data = pd.read_csv(cena)
    z = tuple (data.head(0))    
       
    datas.append(data[z[1]].tolist())
    datas.append(data[z[2]].tolist())
    datas.append(data[z[3]].tolist())
    datas.append(data[z[4]].tolist())
    datas.append(data[z[5]].tolist())
    datas.append(data[z[6]].tolist())        
        
    ponto = ponto+1        

formatado=pd.DataFrame(datas)


n = 0

cab = list(formatado.head(0))

imagens = []

for i in cab:    
    
    x3 = np.array(formatado[i].tolist()) 
    d = 0
    
    blue1  = []
    green1 = []
    red1   = []
    nir1   = []
    swir11 = []
    swir21 = []
    ndvi1  = []
    evi1   = []
    ndwi1  = []
      
    while d < formatado.shape[0]:
        
        blue1.append       (x3[d]/10000)
        green1.append  (x3[d + 1]/10000)
        red1.append    (x3[d + 2]/10000)
        nir1.append    (x3[d + 3]/10000)
        swir11.append  (x3[d + 4]/10000)
        swir21.append  (x3[d + 5]/10000)
        evi1.append (2.5*(x3[d + 3]/10000 - x3[d + 2]/10000)/(x3[d + 3]/10000
                                     +6*x3[d + 2]/10000 - 7.5*x3[d]/10000 +1))
        ndvi1.append  ((x3[d+3]- x3[d+2])/(x3[d+3]+x3[d+2]))
        ndwi1.append ((x3[d+3]- x3[d+4])/(x3[d+3]+x3[d+4]))
        d = d + 6
    
    imagens.append (blue1)
    imagens.append(green1)
    imagens.append  (red1)
    imagens.append  (nir1)
    imagens.append(swir11)
    imagens.append(swir21)
    imagens.append (ndvi1)
    imagens.append  (evi1)
    imagens.append (ndwi1)
    
    n = n + 1

imagens = pd.DataFrame(imagens)    
imagens = np.transpose(imagens)

score14  =  []
mse14    =  []
mae14    =  []

score15  =  []
mse15    =  []
mae15    =  []

   
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
n = 0
cab=tuple(imagens.head(1)) 
t1 = imagens.shape[1]   
re = 0
resuls = []
ano2014 = []
ano2015 = []

ordem   = []
w= 1

while n < t1:    
    
    v = cab [n]
    y =np.array(imagens[v].tolist()) 
    x =np.array(bandas['ID'].tolist()) 
    
    if w >9:
        w =1
            
    x = x[21:35]
    y = y[21:35]
    
    p4 = np.polyfit(x,y,4)
    yfit = p4[0] * pow(x,4) + p4[1] * pow(x,3) + p4[2] * pow(x,2) + p4[3] * x + p4[4]        
    
    score14.append(r2_score(y,yfit))
    mae14.append(mean_absolute_error (y,yfit))
    resul = []
    resul1 =[]
    te =1
    
    while te < 170:
        i = te
        resul.append(p4[0] * pow(i,4) + p4[1] * pow(i,3) + p4[2] * pow(i,2) + p4[3] * i + p4[4])
        te = te + 1
        
    ano2014.append(resul)
    te = 1

    v = cab [n]
    y =np.array(imagens[v].tolist()) 
    x =np.array(bandas['ID'].tolist()) 
    x = x[49:63]
    y = y[49:63]
        
    p4 = np.polyfit(x,y,4)
    yfit = p4[0] * pow(x,4) + p4[1] * pow(x,3) + p4[2] * pow(x,2) + p4[3] * x + p4[4]
            
    score15.append(r2_score(y,yfit))
    mae15.append(mean_absolute_error (y,yfit))
    resul = []
    resul1 =[]
    
    while te < 170:
        i = te      
        resul.append(p4[0] * pow(i,4) + p4[1] * pow(i,3) + p4[2] * pow(i,2) + p4[3] * i + p4[4])
        te = te + 1
         
    ordem.append(w)        
    ano2015.append(resul)
        
    n = n + 1
    w = w + 1

ano2014 = pd.DataFrame(ano2014)
ano2015 = pd.DataFrame(ano2015) 
ano2014['ordem'] = ordem
ano2015['ordem'] = ordem

ano2014 = ano2014.sort_values('ordem')
ano2015 = ano2015.sort_values('ordem')

ano2014 = ano2014.iloc[258:301,:]
ano2015 = ano2015.iloc[258:301,:]

ano2014 = np.transpose(ano2014)
ano2015 = np.transpose(ano2015)

"fator de compensacao de crescimento foliar cl"
import math

n=0
t2 = ano2014.shape[1]
cab2 = list(ano2014.head(0))
minimo = []
maximo= []

while n < t2:    
    
    y = ano2014.iloc[0:169,n:(n+1)]
    mini = np.array(y.min())
    minimo.append(mini[0])
    maxi = (np.array(y.max()))
    maximo.append(maxi[0]*1.1)
    n = n + 1

n = 0
cl= []
t1 = ano2014.shape[0]


while n < t2:    
    
    y = np.array(ano2014.iloc[:,n:n+1].values)
    c = 0
    riaf= []
    
    while c < t1:
        
        fcor = 1 - ((maximo[n]-np.array(y[c][0]))/(maximo[n]-minimo[n]))**0.6
        iaf = -2*math.log(1-fcor)
        clc= 0.515 - math.exp(-0.664-(0.515*iaf))
        riaf.append(clc)
        c = c + 1
    cl.append(riaf)
    n = n + 1 
    
fatorcl=pd.DataFrame(cl)     
   

"producao de materia seca bruta Yo"

import pandas as pd
import math
import numpy as np

ini = 1072
fim = 1283

evapo1 = (evapo.iloc[ini:fim,:])

cab3 = tuple(evapo1.head(1))
t3 = int (evapo1.shape[0])
dia = np.array(evapo1[cab3[17]].tolist())
rse = np.array(evapo1[cab3[14]].tolist())
yo = np.array(evapo1[cab3[15]].tolist())
yc = np.array(evapo1[cab3[16]].tolist())
ky = np.array(evapo1[cab3[18]].tolist())
eto = np.array(evapo1[cab3[10]].tolist())
etc = np.array(evapo1[cab3[11]].tolist())
lat = -16.09
alt = 935

materiabruta=[]
n = 0
for i in dia:
    dsolar = 0.4093*math.sin(2*(math.pi/365)*i-1.405)
    angsolar = math.acos(math.sin(lat)*math.sin(dsolar)-math.cos(lat)*math.cos(dsolar)*math.cos(math.pi*15/180*12))
    azisolar = math.acos((math.cos(lat)*math.sin(dsolar)+math.sin(lat)*math.cos(dsolar)*math.cos(15*12))/math.cos(angsolar))
    dtsolar= 1/ (1+0.033*math.cos(i*2*math.pi/365))
    ondc = ((1367*math.cos(azisolar)*0.75 + 0.000002*alt)/dtsolar)*0.000023885    
    f = (rse[n]-0.5*ondc)/(0.8*rse[n])
    cn = f*yo[n]+(1-f)*yc[n]
    n = n +1
    materiabruta.append(cn)
    
"fator de colheira ch"
ch = 0.10

"fator de producao liquida de materia seca cN"

temp = np.array(evapo1[cab3[1]].tolist())
cn = []

for i in temp:
    if i <20:
        cn.append(0.6)
    if i >20:
        cn.append(0.5)
        
cn.append(0.5)
"produtividade potencial"

"ypi = cl.cn.ch.g.yo"


cab4 = tuple(amostra.head(1))
t4 = int (amostra.shape[0])

ordem = np.array(amostra[cab4[0]].tolist())
prod13= np.array(amostra[cab4[1]].tolist())
prod14= np.array(amostra[cab4[2]].tolist())
prod15= np.array(amostra[cab4[3]].tolist())

n = 0
cab5 = tuple(fatorcl.head(1))
yp = []

for i in ordem:
    cl1= np.array(fatorcl[cab5[i]].tolist())
    antes= prod14[n]    
    v = 0
    prod = []
    
    for j in cl1:
        
        ypi = j*cn[v]*ch*materiabruta[v]*1 
        est= ypi*(1-(1-0.5*(antes/ypi))*(1-ky[v]*(1 - eto[v]/etc[v])))
        prod.append(est)        
        v = v + 1
    n = n + 1
    yp.append(sum(prod)/100)
    
final=pd.DataFrame(yp)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

maeRF = mean_absolute_error (prod15,yp)

scoreRF = r2_score(prod15,yp)   
        
  
    
    


