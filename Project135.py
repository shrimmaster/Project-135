import pandas as pd
import csv
import csv
import pandas as pd

file1 = 'bright_stars.csv'
file2 = 'unit_converted_stars.csv'

d1 = []
d2 = []
with open(file1,'r',encoding='utf8') as f:
    csv_reader =csv.reader(f)
    
    for i in csv_reader:
        d1.append(i)
        
with open(file2,'r',encoding='utf8') as f:
    csv_reader = csv.reader(f)
    
    for i in csv_reader:
        d2.append(i)

h1 = d1[0]
h2 = d2[0]

p_d1 = d1[1:]
p_d2 = d2[1:]

h = h1+h2

p_d =[]

for i in p_d1:
    p_d.append(i)
for j in p_d2:
    p_d.append(j)
with open("total_stars.csv",'w',encoding='utf8') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(h)   
    csvwriter.writerows(p_d)
    
df = pd.read_csv('total_stars.csv')
df.tail(8)
df.columns
df.drop(['Unnamed: 0','Unnamed: 6', 'Star_name.1', 'Distance.1', 'Mass.1', 'Radius.1','Luminosity'],axis=1,inplace=True)
final_data = df.dropna()
print(final_data)

final_data.reset_index(drop=True,inplace = True)

final_data.to_csv('final_data.csv')
final_data.info()
final_data.describe()
final_data.head(5)
final_data.dtypes
print(final_data)

import pandas as pd
df = pd.read_csv('final_data.csv')
df.head()

df.columns
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.head()

dtype: object
df['Radius']=df['Radius'].apply(lambda x: x.replace('$', '').replace(',', '')).astype('float')
radius = df['Radius'].to_list()
mass = df['Mass'].to_list()
gravity =[]

def convert_to_si(radius,mass):
    for i in range(0,len(radius)-1):
        radius[i] = radius[i]*6.957e+8
        mass[i] = mass[i]*1.989e+30
        
convert_to_si(radius,mass)

def gravity_calculation(radius,mass):
    G = 6.674e-11
    for index in range(0,len(mass)):
        g= (mass[index]*G)/((radius[index])**2)
        gravity.append(g)
        
gravity_calculation(radius,mass)

df["Gravity"] = gravity
df

df.to_csv("star_with_gravity.csv")
df.dtypes

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("star_with_gravity.csv")
df.head()

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()
mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius,mass)
#plt.plot(radius,gravity)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass,gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius,mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

from wsgiref import headers
import pandas as pd
import matplotlib.pyplot as plt


print(headers)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns 

x=[]
for index,planet_mass in enumerate(mass):
  temp_list=[
             radius[index],
             planet_mass
             ]
  x.append(temp_list)

wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker='o',color='red')

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()












