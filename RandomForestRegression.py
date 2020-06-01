
import pandas as pd
import numpy as np

# import the order of dates from google earth engine
bandas = pd.read_csv('ordem.csv')

# import productivity sample points
amostra = pd.read_csv('prodcarmo.csv')

# organization of the time series of landsat radiometric values 
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
areas14  =  []
score15  =  []
mse15    =  []
mae15    =  []
areas15  =  []
   
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
ima2014 = []
ima2015 = []
ordem   = []
w= 1

# landsat band polynomial return
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
        
    areas14.append(p4[0] * pow(170,5) + p4[1] * pow(170,4) + p4[2] * pow(170,3)
                                      + p4[3] * pow(170,2) + p4[4] * pow(170,1))   
    
    score14.append(r2_score(y,yfit))
    mae14.append(mean_absolute_error (y,yfit))
    resul = []
    resul1 =[]
    
    for i in x:
        resul.append(p4[0] * pow(i,4) + p4[1] * pow(i,3) + p4[2] * pow(i,2) + p4[3] * i + p4[4])
        
    ano2014.append(resul)
    te = 1

    v = cab [n]
    y =np.array(imagens[v].tolist()) 
    x =np.array(bandas['ID'].tolist()) 
    x = x[49:63]
    y = y[49:63]
        
    p4 = np.polyfit(x,y,4)
    yfit = p4[0] * pow(x,4) + p4[1] * pow(x,3) + p4[2] * pow(x,2) + p4[3] * x + p4[4]
            
    areas15.append(p4[0] * pow(170,5) + p4[1] * pow(170,4) + p4[2] * pow(170,3)
                                      + p4[3] * pow(170,2) + p4[4] * pow(170,1))
    score15.append(r2_score(y,yfit))
    mae15.append(mean_absolute_error (y,yfit))
    resul = []
    resul1 =[]

    for i in x:        
         resul.append(p4[0] * pow(i,4) + p4[1] * pow(i,3) + p4[2] * pow(i,2) + p4[3] * i + p4[4])
         
    ordem.append(w)        
    ano2015.append(resul)
        
    n = n + 1
    w = w + 1

total = ima2014 + ima2015
ordem1 = ordem + ordem

ano2014 = pd.DataFrame(ano2014)
ano2015 = pd.DataFrame(ano2015) 
ano2014['ordem']= ordem
ano2015['ordem']= ordem

ima2014 = pd.DataFrame(ima2014)
ima2015 = pd.DataFrame(ima2015) 
ima2014['ordem']= ordem
ima2015['ordem']= ordem

total = pd.DataFrame(total)
total['ordem'] = ordem1
total = total.sort_values('ordem')

ano2014 = ano2014.sort_values('ordem')
ano2015 = ano2015.sort_values('ordem')

ima2014 = ima2014.sort_values('ordem')
ima2015 = ima2015.sort_values('ordem')

poli15 = []
ima15 = []
n = 0

# correlation analysis 
while n < ano2015.shape[1]:
    
    a = 0
    b = 43
    
    while a < ano2015.shape[0]:
        
        correl = ano2015.iloc[a:b,n:(n+1)]
        x = amostra.iloc[43:86,3:4].values    
        correl['prod'] = x         
        correl3 = correl.corr()
        re1 = np.array( correl3.iloc[0:1,1:2].values)
        re1 = re1[0][0]       
        poli15.append(re1)
        
        a = a + 43
        b = b + 43
        
    n = n + 1    

poli14 = []
ima14 = []
n = 0

while n < ano2014.shape[1]:
    
    a = 0
    b = 43
    
    while a < ano2014.shape[0]:
        
        correl = ano2014.iloc[a:b,n:(n+1)]
        x = amostra.iloc[0:43,3:4].values    
        correl['prod'] = x         
        correl3 = correl.corr()

        re1 = np.array(correl3.iloc[0:1,1:2].values)
        re1 = re1[0][0]
        
        poli14.append(re1)
        
        a = a + 43
        b = b + 43
        
    n = n + 1

tot14 = []
total14 = []
n = 0

while n < total.shape[1]:
    
    a = 0
    b = 86
    
    while a < total.shape[0]:
        
        correl = total.iloc[a:b,n:(n+1)]
        x = amostra.iloc[:,3:4].values    
        correl['prod'] = x        
        correl3 = correl.corr()
        correl4 = correl.corr()
        re1 = np.array(correl3.iloc[0:1,1:2].values)
        re1 = re1[0][0]
        tot14.append(re1)        
        a = a + 86
        b = b + 86
        
    n = n + 1
    
x1 = areas14 + areas15

a  = 0
n  = 0
x2 = []
x3 = []

while n < len(x1):    
       
    x3.append(x1[n])
    
    if a == 8:
        x2.append(x3)
        x3 = []
        a = -1

    n = n + 1
    a = a + 1 
    
x2 = pd.DataFrame(x2)
x = []
x = x2.iloc[:,:]

area_correl = x
area_correl['depois'] = amostra.iloc[:,3:4] 
area_cor = area_correl.corr()

x = x.iloc[:,0:9].values
y = amostra.iloc[:, 3:4].values 

#application of the multiple and random Forest return
from sklearn.ensemble import RandomForestRegressor
regressorRF = RandomForestRegressor(n_estimators=500, 
                                    criterion='mae',random_state = 0)

from sklearn.model_selection import LeaveOneOut
loocv = LeaveOneOut()

previsaoRF = []

for train_index, test_index in loocv.split(x):
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]     

    regressorRF.fit(x_train, y_train)    
    prevRF = regressorRF.predict(x_test)     
    previsaoRF.append(prevRF[0])

#model matching evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

maeRF = mean_absolute_error (y,previsaoRF)
scoreRF = r2_score(y,previsaoRF)
mseRF = mean_squared_error (y,previsaoRF)

imp = regressorRF.feature_importances_

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.subplots()

labels = ['2014','2015']
ax.scatter(previsaoRF[0:43], y[0:43], c = 'blue',label = '2014')
ax.scatter(previsaoRF[43:86], y[43:86], c = 'red',label='2015')
ax.set(xlabel = 'Estimated (sc /ha)',ylabel= 'Observed (sc / ha)',
                 title ='Yield from Random Forest',label = True)

fig.savefig("Yield.jpg", dpi = 600)
plt.show()

index1 = ['Blue', 'Green','Red','Nir','Swir1','Swir2','NDVI','EVI','NDWI']


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['Blue', 'Green','Red','Nir','Swir1','Swir2','NDVI','EVI','NDWI']
sizes = regressorRF.feature_importances_
explode = (0.1, 0, 0, 0,0,0,0,0,0) 
colors = ['blue','green','red','gold', 'yellowgreen', 'lightcoral', 'lightskyblue','gray','orange']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors= colors, autopct='%1.1f%%',
        shadow=False, startangle=90)
ax1.axis('equal') 

fig1.savefig("Importance.jpg",dpi = 600)

plt.show()

a= 0
n = 0
x2 = []
x3 = []

while n < len(poli14):    
    
    if poli14[n]>-0.25 and poli14[n]<0.25:
        x3.append(0)
    else:
        x3.append(poli14[n])
        
    if a == 8:
        x2.append(x3)
        x3 = []
        a = -1

    n = n + 1
    a = a + 1 

correl14 = pd.DataFrame(x2)

a= 0
n = 0
x2 = []
x3 = []

while n < len(poli15):    
    
    if poli14[n]>-0.25 and poli15[n]<0.25:
        x3.append(0)
    else:
        x3.append(poli15[n])
        
    if a == 8:
        x2.append(x3)
        x3 = []
        a = -1

    n = n + 1
    a = a + 1 

correl15 = pd.DataFrame(x2)

a= 0
n = 0

xm14 = []
xm15 = []
xr14 = []
xr15 = []

ym14 = []
ym15 = []
yr14 = []
yr15 = []

while n < len(mae14):
    
    xm14.append(mae14[n])
    xm15.append(mae15[n])
    
    xr14.append(score14[n])
    xr15.append(score15[n])  
            
    if a == 8:
        
        ym14.append(xm14)
        ym15.append(xm15)
        
        yr14.append(xr14)
        yr15.append(xr15)
        
        xm14 = []
        xm15 = []
        xr14 = []
        xr15 = []
        
        a = -1

    n = n + 1
    a = a + 1 

mae15 = pd.DataFrame(ym15)
mae14 = pd.DataFrame(ym14)

score15 = pd.DataFrame(yr15)
score14 = pd.DataFrame(yr14)

import seaborn as sns
import matplotlib.pyplot as plt

index1 = ['Blue', 'Green','Red','Nir','Swir1','Swir2','NDVI','EVI','NDWI']
columns1 = np.array(bandas['ID'].tolist()) 

plot1 =pd.DataFrame(np.transpose(correl14.iloc[0:14,:].values), index=index1 ,columns=columns1[21:35])


fig, ax = plt.subplots (figsize=(9,3))

sns.heatmap(plot1,cmap = 'Spectral_r', vmin=-0.5,vmax=0.5, 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)

plt.savefig ('correl14.jpg', format = 'jpg', dpi = 600)  


plot1 =pd.DataFrame(np.transpose(correl15.iloc[0:14,:].values), index=index1 ,columns=columns1[21:35])


fig, ax = plt.subplots (figsize=(9,3))

sns.heatmap(plot1,cmap = 'Spectral_r', vmin=-0.5,vmax=0.5, 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)

plt.savefig ('correl15.jpg', format = 'jpg', dpi = 600)  


plot1 =pd.DataFrame(np.transpose(score14.iloc[:,:].values), index=index1 )

fig, ax = plt.subplots (figsize=(12,3))

sns.heatmap(plot1,cmap = 'Spectral_r',vmin=0,vmax=1, 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)

plt.savefig ('score14.jpg', format = 'jpg', dpi = 600)  


plot1 =pd.DataFrame(np.transpose(score15.iloc[:,:].values), index=index1 )

fig, ax = plt.subplots (figsize=(12,3))

sns.heatmap(plot1,cmap = 'Spectral_r', vmin=0,vmax=1, 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)

plt.savefig ('score15.jpg', format = 'jpg', dpi = 600)  


plot1 =pd.DataFrame(np.transpose(mae14.iloc[:,:].values), index=index1 )

fig, ax = plt.subplots (figsize=(12,3))

sns.heatmap(plot1,cmap = 'Spectral_r', 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)


plt.savefig ('mae14.jpg', format = 'jpg', dpi = 600)  

plot1 =pd.DataFrame(np.transpose(mae15.iloc[:,:].values), index=index1 )

fig, ax = plt.subplots (figsize=(12,3))

sns.heatmap(plot1,cmap = 'Spectral_r', 
            linecolor='white', linewidths=2, cbar= True, 
            yticklabels = True, xticklabels = True)

plt.savefig ('mae15.jpg', format = 'jpg', dpi = 600) 





