#!/usr/bin/env python
# coding: utf-8

# In[1]:



from collections import defaultdict
import os    
import pandas as pd

uniqueDict=defaultdict(list)
dicPos=defaultdict(int)
dicNeg=defaultdict(int)

#for positive data
filesPos = os.listdir("./pos") 
print("lehgth of total files in positive is",len(filesPos),". So training data is till 600.");print()

for i in range(600):
    file=filesPos[i].split('.')
    if file[0] != "":
        data=open("./pos/"+file[0]+'.txt',encoding="utf-8",errors='ignore' )    
        df=data.read()

        for word in df.split():
            if word not in uniqueDict:
                uniqueDict[word]
            if word not in dicPos:
                dicPos[word]=1
            else:
                dicPos[word]+=1

#for negative data
filesNeg = os.listdir("./neg") 
print("lehgth of total files in negative is",len(filesNeg),". So training data is till 600.");print()

for i in range(600):
    file=filesNeg[i].split('.')
    if file[0] != "":
        data=open("./neg/"+file[0]+'.txt',encoding="utf-8",errors='ignore' )    
        df=data.read()

        for word in df.split():
            if word not in uniqueDict:
                uniqueDict[word]
            if word not in dicNeg:
                dicNeg[word]=1
            else:
                dicNeg[word]+=1
                
                
vocab=len(uniqueDict)
pos=len(dicPos)
neg=len(dicNeg)              
print("words in +ve",pos,"; in -ve",neg,"and total unique words",vocab);print()

# Naive Base training for 1200 documents (600 +ve and 600 -ve)
probPos=defaultdict(float)
probNeg=defaultdict(float)

totalPos=0;totalNeg=0    #total counts of duplicates words
for word in dicPos:
    totalPos+=dicPos[word]

for word in dicNeg:
    totalNeg+=dicNeg[word]

for word in dicPos:   #probability of a word in both +ve and -ve 
    probPos[word] = (1+dicPos[word])/(vocab+totalPos)     # plus 1, in case of count=0
for word in dicNeg:
    probNeg[word] = (1+dicNeg[word])/(vocab+totalNeg)


# In[2]:


#testing +ve the data remaining 93 files in both +ve and -ve
import math
count=0
for i in range(600,693,1):
    file=filesPos[i].split('.')
    if file[0] != "":
        data=open("./pos/"+file[0]+'.txt',encoding="utf-8",errors='ignore' )    
        df=data.read()

#         print(file[0]+"."+file[1],end=" is ")
        
        dic=defaultdict(int)
        for word in df.split():
            dic[word]= 1 if word not in dic else (dic[word]+1)
        
        positive=0.5    
        negative=0.5
        notFound=10**(-20)
        for word in dic:   #probability of a word in both +ve and -ve
            positive += math.log10(probPos[word]) if word in probPos else math.log10(notFound)
            negative += math.log10(probNeg[word]) if word in probNeg else math.log10(notFound)

        if (positive >negative):
            count+=1
#             print("+ve")
#         else:
#             print("-ve")
        


# In[3]:


#testing -ve the data remaining 
import math
countN=0
for i in range(600,693,1):
    file=filesNeg[i].split('.')
    if file[0] != "":
        data=open("./neg/"+file[0]+'.txt',encoding="utf-8",errors='ignore' )    
        df=data.read()

#         print(file[0]+"."+file[1],end=" is ")
        
        dic=defaultdict(int)
        for word in df.split():
            dic[word]= 1 if word not in dic else (dic[word]+1)
        
        positive=0.5    
        negative=0.5
        notFound=10**(-20)
        for word in dic:   #probability of a word in both +ve and -ve
            
            positive += math.log10(probPos[word]) if word in probPos else math.log10(notFound)
            negative += math.log10(probNeg[word]) if word in probNeg else math.log10(notFound)
#             if word in probPos:
#                 positive *= probPos[word]
#             if word in probNeg:
#                 negative *= probNeg[word]
#             print(positive,negative)
    
        if (positive < negative):
            countN+=1
#             print("-ve")
#         else:
#             print("+ve")
        


# In[4]:


acc=(count)*100/93
accN=countN*100/93
print("accuracy for negative 93 files",accN)
print("accuracy for positive 93 files",acc)


# In[ ]:




