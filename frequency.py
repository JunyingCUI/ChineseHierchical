# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:44:04 2022

@author: Administrator
"""
import pandas as pd




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
    
    dir_to_read = "Frequency"
    #texts=["家春秋.txt"]
    texts = ["祝福.txt"]
    #texts = ["太阳照在桑干河上.txt", "酒徒刘以鬯.txt", "祝福.txt", "射雕英雄传 金庸 1.txt","家春秋.txt"]
    data=[]   
    for title in texts:
        sentence=readText(title)
        
    d={char:sentence.count(char) for char in set(sentence) if "u4e00"<=char<="\u9fff"}
    data=sorted(d.items(),key=lambda x:x[1],reverse=True)
    
    
    delta_alpha_df = pd.DataFrame(data, columns=["Title", "Samples"])
    delta_alpha_df.to_csv(dir_to_read + ".csv", sep='\t')