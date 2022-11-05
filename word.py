# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:13:03 2022

@author: Administrator
"""

import pandas as pd
import re
import collections
from zhon.hanzi import punctuation
import numpy as np
from numpy import linalg as la

# when given the name of a text file excluding the extension, this function reads the text and returns a string
def readText(filename):
    text = ""
    try:
        f = open(filename, "r", encoding='utf-8')
        text = f.read()
        f.close()
    
    except UnicodeDecodeError:
        f = open(filename, "r", encoding='utf-16')
        text = f.read()
        f.close()
    return text



if __name__=='__main__':
    
    dir_to_read = "word-vector"
    #texts=["家春秋.txt"]
    texts = ["祝福.txt"]
    #texts = ["太阳照在桑干河上.txt", "酒徒刘以鬯.txt", "祝福.txt", "射雕英雄传 金庸 1.txt","家春秋.txt"]
    data=[]   
    for title in texts:
        sentence=readText(title)
        
        
        #punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……￥·"""
        '''  
        punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~“”？，！【】（）、。：；’‘……\n\t￥·"""
        for i in punctuation:
            sentence=sentence.replace(i, "")   #清除中文标点符号
        '''
        

        for i in punctuation:
            sentence=sentence.replace(i, "")   #清除中文标点符号
        sentence=sentence.replace('\n', "")   #清除换行符
        #counter = collections.Counter(sentence)
        #对每个字  从出现的次数  从小到大的排序
        #counter_pairs = sorted(counter.items(), key=lambda x:-x[1])
        #print(counter_pairs)                              
         
        #对字和出现的次数进行分割
        #words,_ = zip(*counter_pairs) 
        
        '''
        W=np.zeros((len(words),len(sentence)))
        for i in range(len(words)):
            for j in range(len(sentence)):
                if words[i]==sentence[j]:
                    W[i][j]=1
        U,sigma,VT=la.svd(W)
        print(sigma)
        delta_alpha_df = pd.DataFrame(counter_pairs,sigma)
        delta_alpha_df.to_csv(dir_to_read + ".csv", sep='\t')        
       '''                              
        
        
        a=200  #窗口的长度
        S=np.zeros((a,len(sentence)))   
        for i in range(len(sentence)):
            subsentence=[]
            for j in range(a):
                if((i+j)<len(sentence)):            
                    subsentence.append(sentence[i+j])                   
            counter = collections.Counter(subsentence)
            counter_pairs = sorted(counter.items(), key=lambda x:-x[1])                              
            #对字和出现的次数进行分割
            words,_ = zip(*counter_pairs) 
            W=np.zeros((len(words),len(subsentence)))
            for i in range(len(words)):
                for j in range(len(subsentence)):
                    if words[i]==subsentence[j]:
                        W[i][j]=1
            U,sigma,VT=la.svd(W)
            #print(len(sigma))           
            #S.append(sigma)
            #print(sigma)
            delta_alpha_df = pd.DataFrame(sigma)
            delta_alpha_df.to_csv(dir_to_read + ".csv", sep='\t')    
        
            

            
        
        
       
        
        
        '''
        rep = {'\n':'',' ':''}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        origin_words = pattern.sub(lambda m: rep[re.escape(m.group(0))], sentence)
        
        #print(origin_words)
        
        d={char:sentence.count(char) for char in set(sentence) if "u4e00"<=char<="\u9fff"}
        #data=sorted(d.items(),key=lambda x:x[1],reverse=True)
        data=sorted(d.items(),key=lambda x:x[1],reverse=True)
    
    
    
        #delta_alpha_df = pd.DataFrame(data, columns=["Title", "Samples"])
        delta_alpha_df = pd.DataFrame(data, columns=["Title", "Samples"])
        delta_alpha_df.to_csv(dir_to_read + ".csv", sep='\t')
        ''' 