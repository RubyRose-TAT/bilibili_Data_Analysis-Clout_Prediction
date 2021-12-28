# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 08:34:35 2021

@author: Administrator
"""
import time
import requests
from lxml import etree
import json
import csv
import re
from datetime import datetime
import numpy as np
import pandas as pd
import random


def dataprocess(a):
    a_str = ' '.join(str(i) for i in a)
    b = a_str.strip()
    return b
        

def getfans(url,headers):
    url_mid = 'https://api.bilibili.com/x/web-interface/card?mid='+str(url) 
    page_text = requests.get(url_mid,headers).text
    fans_data = json.loads(page_text)
    fans = fans_data['data']['follower']
    return fans

   #快速爬取，获取三个元素只需解析一次页面，返回包含三个元素的列表
def get_list_uploadtime_channel_lables(str_bvid,headers):
    data_list=[]
    url = 'https://www.bilibili.com/video/'+str_bvid+'?spm_id_from=333.934.0.0'
    page_text = requests.get(url=url, headers = headers).text
    tree = etree.HTML(page_text)
    uploadtime = tree.xpath('//*[@id="viewbox_report"]/div/span[3]/text()')
    str_uploadtime = dataprocess(uploadtime)
    channel = tree.xpath('//*[@id="v_tag"]/ul/li/div/a/span/text()')
    str_channel = dataprocess(channel)
    lable = tree.xpath('//*[@id="v_tag"]/ul/li/a/span/text()')
    str_lable = dataprocess(lable)
    data_list.append(str_uploadtime)
    data_list.append(str_channel)
    data_list.append(str_lable)
    return data_list

def getbilibilidata(headers,periodical_number):
   data={}
   data['author'] = mid_data['data']['list'][i]['owner']['name']
   mid = mid_data['data']['list'][i]['owner']['mid']
   follower = getfans(mid, headers)
   data['follower'] = follower
   data['title'] = mid_data['data']['list'][i]['title']
   data['like'] =mid_data['data']['list'][i]['stat']['like'] 
   data['coin'] = mid_data['data']['list'][i]['stat']['coin']
   data['collect'] = mid_data['data']['list'][i]['stat']['favorite']    
   data['share'] = mid_data['data']['list'][i]['stat']['share']
   data['danmaku'] = mid_data['data']['list'][i]['stat']['danmaku']
   data['view'] = mid_data['data']['list'][i]['stat']['view']
   data['reply'] = mid_data['data']['list'][i]['stat']['reply']
   bvid = mid_data['data']['list'][i]['bvid']
   uploadtime_channel_lables_list=get_list_uploadtime_channel_lables(bvid,headers)
   data['uploadtime'] = uploadtime_channel_lables_list[0]
   data['capturetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   data['channel'] = uploadtime_channel_lables_list[1]
   data['lables'] = uploadtime_channel_lables_list[2]
   data['periodical_number'] = str(periodical_number)
   return data
if __name__ == "__main__":  
    #UA伪装
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7',
        'Connection': 'keep-alive',
        #'Host': 'api.bilibili.com',
        'Origin': 'https://www.bilibili.com',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.5'
        }
    index_id = 0 
    for i in range(140,140):
        url = 'https://api.bilibili.com/x/web-interface/popular/series/one?number='+str(i)
        page_text = requests.get(url=url, headers = headers).text
        mid_data = json.loads(page_text)
        periodical_number = i
        for i in range(0,len(mid_data['data']['list'])):
            data = getbilibilidata(headers,periodical_number)
            df = pd.DataFrame(data,index = [index_id])
            df.to_csv('bilibili_week_popular140.csv',encoding="utf_8_sig",header=0,mode='a',na_rep='NA')
            index_id += 1
            print(data)
        sleeptime = random.randint(20,30) 
        time.sleep(sleeptime)
            
    

#慢速爬取，一个一个页面请求，速度特别慢    
# def getuploadtime(str_bvid,headers):
#     url = 'https://www.bilibili.com/video/'+str_bvid+'?spm_id_from=333.934.0.0'
#     page_text = requests.get(url=url, headers = headers).text
#     tree = etree.HTML(page_text)
#     uploadtime = tree.xpath('//*[@id="viewbox_report"]/div/span[3]/text()')
#     str_uploadtime = dataprocess(uploadtime)
#     return str_uploadtime

# def getchannel(str_bvid,headers):
#     url = 'https://www.bilibili.com/video/'+str_bvid+'?spm_id_from=333.934.0.0'
#     page_text = requests.get(url=url, headers = headers).text
#     tree = etree.HTML(page_text)
#     channel = tree.xpath('//*[@id="v_tag"]/ul/li/div/a/span/text()')
#     str_channel = dataprocess(channel)
#     return str_channel

# def getlables(str_bvid,headers):
#     url = 'https://www.bilibili.com/video/'+str_bvid+'?spm_id_from=333.934.0.0'
#     page_text = requests.get(url=url, headers = headers).text
#     tree = etree.HTML(page_text)
#     lable = tree.xpath('//*[@id="v_tag"]/ul/li/a/span/text()')
#     str_lable = dataprocess(lable)
#     return str_lable
        
        # data['uploadtime'] = getuploadtime(bvid, headers)
        # data['channel'] = getchannel(bvid, headers)
        # data['lables'] = getlables(bvid, headers) 
 
    # url = 'https://api.bilibili.com/x/web-interface/popular/series/one?number=128'
    # page_text = requests.get(url=url, headers = headers).text
    # mid_data = json.loads(page_text)
    # for i in range(0,len(mid_data['data']['list'])):
    #     data = getbilibilidata(headers)
    #     df = pd.DataFrame(data,index = [i])
    #     df.to_csv('bilibili2.csv',encoding="utf_8_sig",header=0,mode='a',na_rep='NA')
    #     i+=1
    #     print(data)  
    
    
    
    
       
    
    
    
    
    
    
    
    
    
    
    