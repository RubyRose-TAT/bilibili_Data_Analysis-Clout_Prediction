# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:45:59 2021

@author: Administrator
"""
import time
import requests
from lxml import etree
import json
from datetime import datetime
import pandas as pd
import random

#数据加工，将xpath解析出来的列表数据转化为str
def dataprocess(a):
    a_str = ' '.join(str(i) for i in a)
    b = a_str.strip()
    return b

#获取粉丝数
def getfans(url,headers):
    #先睡眠2秒，防止请求太快被封IP
    time.sleep(4)
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
#获取视频数据
def getbilibilidata(headers,partion_name):
    #每次爬取先睡眠1秒，防止被封ip
    time.sleep(1)
    data={}
    data['author'] = mid_data['data']['owner']['name']
    mid = mid_data['data']['owner']['mid']
    follower = getfans(mid, headers)
    data['follower'] = follower
    data['title'] = mid_data['data']['title']
    data['like'] =mid_data['data']['stat']['like'] 
    data['coin'] = mid_data['data']['stat']['coin']
    data['collect'] = mid_data['data']['stat']['favorite']    
    data['share'] = mid_data['data']['stat']['share']
    data['danmaku'] = mid_data['data']['stat']['danmaku']
    data['view'] = mid_data['data']['stat']['view']
    data['reply'] = mid_data['data']['stat']['reply']
    data['duration'] =mid_data['data']['duration']
    bvid = mid_data['data']['bvid']
    #这部分内容不在接口数据，跳到视频页使用xpath解析出来
    uploadtime_channel_lables_list=get_list_uploadtime_channel_lables(bvid,headers)
    data['uploadtime'] = uploadtime_channel_lables_list[0]
    data['capturetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['channel'] = uploadtime_channel_lables_list[1]
    data['lables'] = uploadtime_channel_lables_list[2]
    data['partion_name'] = str(partion_name)
    return data
if __name__ == "__main__":  
    #UA伪装
    my_headers = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.5'
                ] 
    headers = {
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7',
        'Connection': 'keep-alive',
        'Origin': 'https://www.bilibili.com',
        'User-Agent':random.choice(my_headers)
         }
    partition_list = ['guochuang','douga','music','dance','game','knowledge','tech','sports','car']
    #'all','guochuang','douga','music','dance','game','knowledge','tech','sports','car','life','food','animal','kichiku','fashion','ent','cinephile','origin','rookie'
    for partition_name in partition_list: 
        url='https://www.bilibili.com/v/popular/rank/'+ str(partition_name)
        page_text= requests.get(url, headers).content
        #将页面数据转化为etree对象
        tree = etree.HTML(page_text)
        #指定数据所在li标签的xpath总路径
        li_list = tree.xpath('//*[@id="app"]/div/div[2]/div[2]/ul/li')
        #url_list用来存放各个视频链接
        url_list = []
        #循环遍历总路径获取各个视频的bv号，并将其拼接成视频接口链接，最后append在url_list中
        for li in li_list:
            url = li.xpath('./div[1]/div[1]/a/@href')
            str_url = ' '.join(url)
            str_url = str_url[25:]
            str_url = 'https://api.bilibili.com/x/web-interface/view?bvid='+ str_url
            url_list.append(str_url)
        #设置索引号，作为表格的第一列用于计数
        index_id = 1
        #依次请求各个视频接口，获取数据
        for post_url in url_list:
            page_text = requests.get(url=post_url, headers=headers).text
            #将数据json化，方便数据定位
            mid_data = json.loads(page_text)
            data = getbilibilidata(headers,partition_name)
            #将获得的字典data转化为二维表存起来
            df = pd.DataFrame(data,index = [index_id])
            df.to_csv('bilibili_rank100_data_12_14.csv',encoding="utf_8_sig",header=0,mode='a',na_rep='NA')
            index_id += 1
            print(data)
    sleeptime = random.randint(10,20) 
    time.sleep(sleeptime)
        
    
    
        