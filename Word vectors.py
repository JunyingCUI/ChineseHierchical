# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:56:45 2022

@author: Administrator
"""

import re
import collections
 
words = '''
    钟声响起归家的讯号
    在他生命里
    仿佛带点唏嘘
    黑色肌肤给他的意义
    是一生奉献 肤色斗争中
    年月把拥有变做失去
    疲倦的双眼带着期望
    今天只有残留的躯壳
    迎接光辉岁月
    风雨中抱紧自由
    一生经过彷徨的挣扎
    自信可改变未来
    问谁又能做到
    可否不分肤色的界限
    愿这土地里
    不分你我高低
    缤纷色彩闪出的美丽
    是因它没有
    分开每种色彩
    年月把拥有变做失去
    疲倦的双眼带着期望
    今天只有残留的躯壳
    迎接光辉岁月
    风雨中抱紧自由
    一生经过彷徨的挣扎
    自信可改变未来
    问谁又能做到
    今天只有残留的躯壳
    迎接光辉岁月
    风雨中抱紧自由
    一生经过彷徨的挣扎
    自信可改变未来
    问谁又能做到
    今天只有残留的躯壳
    迎接光辉岁月
    风雨中抱紧自由
    一生经过彷徨的挣扎
    自信可改变未来
    问谁又能做到
    今天只有残留的躯壳
    迎接光辉岁月
    风雨中抱紧自由
    一生经过彷徨的挣扎
    自信可改变未来
'''
 
'替换\n 和空格'
 
 
rep = {'\n':'',' ':''}
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
origin_words = pattern.sub(lambda m: rep[re.escape(m.group(0))], words)
#print(origin_words)
 
counter = collections.Counter(origin_words)
#print(counter)
#Counter({'的': 16, '自': 10, '有': 8, '生': 7, '月': 7, '变': 7, '一': 6, '中': 6, '做': 6, '可': 6, '色': 5, '今': 5, '天': 5, '只': 5, '残': 5, '留': 5, '躯': 5, '壳': 5, '迎': 5, '接': 5, '光': 5, '辉': 5, '岁': 5, '风': 5, '雨': 5, '抱': 5, '紧': 5, '由': 5, '经': 5, '过': 5, '彷': 5, '徨': 5, '挣': 5, '扎': 5, '信': 5, '改': 5, '未': 5, '来': 5, '问': 4, '谁': 4, '又': 4, '能': 4, '到': 4, '带': 3, '肤': 3, '分': 3, '他': 2, '里': 2, '是': 2, '年': 2, '把': 2, '拥': 2, '失': 2, '去': 2, '疲': 2, '倦': 2, '双': 2, '眼': 2, '着': 2, '期': 2, '望': 2, '不': 2, '彩': 2, '钟': 1, '声': 1, '响': 1, '起': 1, '归': 1, '家': 1, '讯': 1, '号': 1, '在': 1, '命': 1, '仿': 1, '佛': 1, '点': 1, '唏': 1, '嘘': 1, '黑': 1, '肌': 1, '给': 1, '意': 1, '义': 1, '奉': 1, '献': 1, '斗': 1, '争': 1, '否': 1, '界': 1, '限': 1, '愿': 1, '这': 1, '土': 1, '地': 1, '你': 1, '我': 1, '高': 1, '低': 1, '缤': 1, '纷': 1, '闪': 1, '出': 1, '美': 1, '丽': 1, '因': 1, '它': 1, '没': 1, '开': 1, '每': 1, '种': 1})
 
 
#对每个字  从出现的次数  从小到大的排序
counter_pairs = sorted(counter.items(), key=lambda x:-x[1])
#print(counter_pairs)                              
 
#对字和出现的次数进行分割
words,_ = zip(*counter_pairs)                                       
#print(words)
 
#对每个字进行编码
word_int_map = dict(zip(words, range(len(words))))
print(word_int_map)
 
 
#对文本构建文本向量
word_vector = [list(map(lambda word: word_int_map.get(word, len(words)), origin_words))]
print(word_vector)