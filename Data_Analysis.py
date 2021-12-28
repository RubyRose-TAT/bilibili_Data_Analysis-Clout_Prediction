import os
import re
import jieba
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from collections import Counter
from streamlit_echarts import st_pyecharts
from streamlit_echarts import JsCode
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.globals import SymbolType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components


def data_analysis():
    st.sidebar.markdown('---')
    image = Image.open('data/2233.jpg')
    st.sidebar.image(image, caption='国风2233')

    st.write("""
    # 数据可视化分析
    """)
    st.write('---')

    #  数据预处理
    df = pd.read_csv('data/bilibili_data_analysis.csv')
    df = df.drop_duplicates(subset=['标题'],keep='first',inplace=False)
    df = df.drop(df[df['时间']>100].index)
    df[df.isnull().values==True]
    df = df.reset_index(drop=True, inplace=False)
    df = df.drop_duplicates(subset=['标题'],keep='last',inplace=False)
    df['点赞/播放比'] = list(map(lambda x,y: x/y, df['点赞'],df['播放']))

    st.write("""
    ## 数据集展示
    """)
    st.write ( '数据集: ' + str(df.shape[0]) + ' 行 ' + str(df.shape[1]) + '列.')
    st.dataframe(df)
    st.caption('数据来源: 自己写的爬虫，爬取[bilibili排行版](https://www.bilibili.com/v/popular/rank/all)，获取各分区排行榜前100视频数据,保存于本地csv文件中')

    st.write('---')
    st.write("""
    ## 标题分析
    """)
    st.write('---')

    # 标题写入txt
    if(os.path.isfile("bilibili_title.txt")):
        os.remove(r'bilibili_title.txt')
    with open('bilibili_title1.txt','w',encoding="utf-8") as fp:
        for title in df['标题']: 
            fp.write(title)
    fp.close()
    # jieba分词
    title_cut_list = []
    with open('bilibili_title1.txt','r',encoding="utf-8") as fp:
        for title in fp:
            title = title.replace("\n", "")
            title_cut = jieba.lcut(title)
            title_cut_list.append(title_cut) 
    fp.close()
    # 设置停用词
    stopwords = set()
    content = [line.strip() for line in open('data/stopwords.txt','r', encoding='utf-8').readlines()]
    stopwords.update(content)
    c= Counter()
    for a in title_cut_list[0]:
        if len(a)>1 and a not in stopwords:
            c[a] += 1
    # 绘制词云图
    word_counts_top200 = c.most_common(200)
    word1 = (
        WordCloud(init_opts=opts.InitOpts(width='1350px', height='750px', theme=ThemeType.MACARONS))
        .add('词频', data_pair=word_counts_top200,
            word_size_range=[15, 108], textstyle_opts=opts.TextStyleOpts(font_family='cursive'),
            shape=SymbolType.DIAMOND)
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="标题词云", subtitle="词频top200"
            ),
            toolbox_opts=opts.ToolboxOpts(),
        )
        .render_embed()
    )
    components.html(word1, width=1350, height=750)

    st.write('---')
    st.write("""
    #####  标题词频柱状图
    """)

    # 对标题关键词进行分析
    x_data = []
    y_data = []
    for i in range(0,60):
        x_data.append(c.most_common(200)[i][0])
        y_data.append(c.most_common(200)[i][1])

    word2 = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
        .add_xaxis(x_data)
        .add_yaxis("出现频次",y_data,label_opts=opts.LabelOpts(is_show=False,position='top'), 
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, 
                            [{
                                offset: 0,
                                color: 'rgb(255,99,71)'
                            }, {
                                offset: 1,
                                color: 'rgb(32,178,170)'
                            }])""")
            )
        )
        .set_global_opts(
        toolbox_opts=opts.ToolboxOpts(),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),
        xaxis_opts=opts.AxisOpts(name='发布时间间隔',
                                                type_='category',                                           
                                                axislabel_opts=opts.LabelOpts(rotate=45),
                                                ),
        yaxis_opts=opts.AxisOpts(name='', splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(type_='dash')),)            
        )
        .render_embed()
    )
    components.html(word2, width=1000, height=500)

    st.write('---')
    st.write("""
    #####  含特殊符号标题分析
    """)

    pattern = "？"
    query_count = len(df[df['标题'].astype(str).str.contains(pattern,regex = True)].sort_values(by=['播放'],ascending=False))
    pattern = "！"
    exclamation_count = len(df[df['标题'].astype(str).str.contains(pattern,regex = True)].sort_values(by=['播放'],ascending=False))
    pattern = "？！|！？"
    query_exclamation_count=len(df[df['标题'].astype(str).str.contains(pattern,regex = True)].sort_values(by=['播放'],ascending=False))
    total_video = len(df)
    other_video = total_video+query_exclamation_count-query_count-exclamation_count

    data_list = [query_count,exclamation_count,other_video ]
    table_list = ['含问号标题','含感叹号标题','两者都不含的标题']

    # 符号饼图
    word3 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(table_list,data_list)],
            radius=["40%", "55%"],
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                background_color="#eee",
                border_color="#aaa",
                border_width=1,
                border_radius=4,
                rich={
                    "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                    "abg": {
                        "backgroundColor": "#e3e3e3",
                        "width": "100%",
                        "align": "right",
                        "height": 22,
                        "borderRadius": [4, 4, 0, 0],
                    },
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 16, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
        .render_embed()
    )
    components.html(word3, width=1000, height=500)

    st.write("""
    #####  结论
    标题这里能简单的到一些信息，人们通常喜欢用一些什么，表示惊讶的句子来当标题，这样好像很吸引人
    """)

    st.write('---')

    st.write("""
    ## 各大分区播放量分析
    """)

    # 数据处理

    df = df.drop(df[df['分区'] == 'all'].index)
    df=df.reset_index(drop=True, inplace=False)
    animal_df = df[df['分区'] == 'animal'].sort_values(by=['播放'],ascending=False)
    animal_df['播放'].mean()
    partition_group_df = df.groupby('分区')
    # 将df中的数据统一保留两位小数
    def data_normalization(a):
        mean_np = np.array(a)
        mean_np_2f = np.round(mean_np,2) 
        return list(mean_np_2f)
    # 求出各大分区的播放量均值和最大值
    view_mean_list1 = list(partition_group_df.agg({'播放':'mean'})['播放'].values/10000)
    view_max_list1 =  list(partition_group_df.agg({'播放':'max'})['播放'].values/10000)
    view_mean_list = data_normalization(view_mean_list1) # 将列表中的均值保留两位小数                
    view_max_list = data_normalization(view_max_list1) # 将列表中的最大值保留两位小数

    partition_list=['animal','car','cinephile','dance','douga','ent','fashion','food','game','guochuang','kichiku','knowledge','life','music','origin','rookie','sports','tech']
    
    st.write('---')
    st.write("""
    #####  各大分区播放量分布
    """)

    c1 = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
        .add_xaxis(partition_list)
        .add_yaxis("各分区播放量均值",view_mean_list)
        .add_yaxis("各分区最大播放量",view_max_list)
        .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                title_opts=opts.TitleOpts(title="", subtitle="单位:万"),
                tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),
                legend_opts=opts.LegendOpts(selected_mode = 'single')
            )
        .render_embed()
    )
    components.html(c1, width=1000, height=500)

    st.write('---')

    st.write("""
    #####  各大分区播放量均值
    """)

    c2 = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(partition_list,view_mean_list)],
            radius=["40%", "75%"],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="6%", pos_left="2%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        .render_embed()
    )
    components.html(c2, width=1000, height=500)

    st.write('---')

    c3 = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
        .add_xaxis(['animal','car','cinephile','dance','douga','ent','fashion','food','game','guochuang','kichiku','knowledge','life','music','origin','rookie','sports','tech'])
        .add_yaxis("各大分区最大播放量",view_max_list)
        .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
                title_opts=opts.TitleOpts(title="B站播放量分布", subtitle="单位:万"),
                tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross')
                )
        .render_embed()
    )
    components.html(c3, width=1000, height=500)
   
    st.write("""
    #####  各大分区播放量结论
    内容补充____zsbd
    """)

    st.write('---')
    st.write("""
    ##  播放量靠前的数据分析
    """)

    st.write('---')
    st.write("""
    ##### top100播放量与发布时间的分析
    """)

    # 数据处理
    head_100_video_sort_by_view_df = df.sort_values(by=['播放'],ascending=False).head(100)
    head_100_video_df = head_100_video_sort_by_view_df.sort_values(by=['时间'],ascending=True).head(100) #按时间排序，准备分析时间长短对播放量的影响，时间作为横轴，播放量为纵轴
    head_100_video_time_list = data_normalization(list(head_100_video_df['时间'].values))
    head_100_video_view_list = data_normalization(list(head_100_video_df['播放'].values/10000))
    head_100_video_like_list = data_normalization(list(head_100_video_df['点赞'].values/1000))
    head_100_video_coin_list = data_normalization(list(head_100_video_df['硬币'].values/1000))

    b1 = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
        .add_xaxis(head_100_video_time_list)
        .add_yaxis("播放量/万",head_100_video_view_list,label_opts=opts.LabelOpts(is_show=False,position='top'), itemstyle_opts=opts.ItemStyleOpts(
                                        color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, 
                                                    [{
                                                        offset: 0,
                                                        color: 'rgb(255,99,71)'
                                                    }, {
                                                        offset: 1,
                                                        color: 'rgb(32,178,170)'
                                                    }])"""))
                        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="时间与播放量关系"),
            toolbox_opts=opts.ToolboxOpts(),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),
            xaxis_opts=opts.AxisOpts(name='发布时间',
                                                    type_='category',                                           
                                                    axislabel_opts=opts.LabelOpts(rotate=45),
                                                    ),
            yaxis_opts=opts.AxisOpts(name='', splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(type_='dash')),)
        )
        .render_embed()
    )
    components.html(b1, width=1000, height=500)

    st.write("""
    #####  结论
    发布时间对视频播放量的影响不是很大，作为创作者应该在视频内容质量上下功夫
    """)

    st.write('---')
    st.write("""
    ##### 发布时间与上热门视频数量
    """)

    k= 0
    #统计各个时间段视频数量
    video_number_list = []
    xlable_time = [] #用作横坐标参数
    for i in range(0,31):
        k=i
        day_to_day = str(k)+'-'+str(k+1)
        xlable_time.append(day_to_day)
        video_number_list.append(len(df[(df['时间'].values<k+1) & (df['时间'].values>=k)]))
    video_number_list.append(len(df[df['时间'].values>=31]))
    xlable_time.append('>=31')

    b2 = (
            Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
            .add_xaxis(xlable_time)
            .add_yaxis("视频数量",video_number_list,label_opts=opts.LabelOpts(is_show=False,position='top'), itemstyle_opts=opts.ItemStyleOpts(
                                            color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, 
                                                        [{
                                                            offset: 0,
                                                            color: 'rgb(255,99,71)'
                                                        }, {
                                                            offset: 1,
                                                            color: 'rgb(32,178,170)'
                                                        }])"""))
                            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=""),
                toolbox_opts=opts.ToolboxOpts(),
                legend_opts=opts.LegendOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),
                xaxis_opts=opts.AxisOpts(name='发布时间',
                                                        type_='category',                                           
                                                        axislabel_opts=opts.LabelOpts(rotate=45),
                                                        ),
                yaxis_opts=opts.AxisOpts(name='', splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(type_='dash')),

                                        )
            )  
            .render_embed()
    )
    components.html(b2, width=1000, height=500)

    st.write("""
    #####  结论
    发布了越久的视频上热门的再次概率是比较小的，除非那个视频真的很火，所以创作者们观察前一两天视频的热度，就可以知道视频制作效果怎么样
    """)

    st.write('---')
    st.write("""
    ##### top20视频，播放量与各大因素的关系
    """)
    
    # 数据处理
    head_100_video_df = head_100_video_sort_by_view_df
    head_100_video_time_list = data_normalization(list(head_100_video_df['时间'].values/24))
    head_100_video_view_list = data_normalization(list(head_100_video_df['播放'].values/10000))
    head_100_video_like_list = data_normalization(list(head_100_video_df['点赞'].values/10000))
    head_100_video_coin_list = data_normalization(list(head_100_video_df['硬币'].values/1000))
    df['点赞/播放比'] = list(map(lambda x,y: x/y, df['点赞'],df['播放']))

    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
    bar.add_xaxis(head_100_video_df['标题'][:20].tolist())
    bar.add_yaxis('播放量/万',y_axis=head_100_video_view_list[:20],
                yaxis_index=0,
                label_opts=opts.LabelOpts(is_show=False),
                stack='stack1',
                color="#d14a61"
                )
    bar.add_yaxis('点赞/万',head_100_video_like_list[:20],
                yaxis_index=0,label_opts=opts.LabelOpts(is_show=False),
                stack='stack1',
                color="#5793f3"
                )
    #extend_axis yaxis而不是yaxis_opts
    bar.extend_axis(yaxis=opts.AxisOpts(name='点赞/播放比',
                                        min_ = -1,
                                        max_ = 0.5,
                                        position='right',

                                        axisline_opts=opts.AxisLineOpts(
                                        linestyle_opts=opts.LineStyleOpts(color="#675bba")
                                            ),
                                        axislabel_opts=opts.LabelOpts(formatter="{value} %")
                                    )
                )
    bar.set_global_opts(title_opts=opts.TitleOpts(title='播放量Top20视频'),
                        xaxis_opts=opts.AxisOpts(
                            name='',
                            type_='category',
                            name_gap = 35,
                            axislabel_opts=opts.LabelOpts(interval=0,rotate=20)),
                        yaxis_opts=opts.AxisOpts(name='',splitline_opts=opts.SplitLineOpts(is_show=True),
                                                axislabel_opts=opts.LabelOpts(formatter="{value}万")
                                                ),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", 
                                                    axis_pointer_type="cross"
                                                    )
                        
                    )
    line=Line()
    line.add_xaxis(head_100_video_df[:20]['标题'])
    line.add_yaxis('点赞/播放比',
                y_axis=head_100_video_df['点赞/播放比'].tolist(),               
                label_opts=opts.LabelOpts(is_show=False),
                symbol='emptyCircle',
                is_symbol_show=True,
                color="#675bba",
                yaxis_index=1
                )
    st_pyecharts(bar.overlap(line),height="500px",width="1000px")

    st.write('---')
    st.write("""
    ##### 播放量top30作品漏斗图
    """)

    list_funnel = ['播放','点赞','硬币','收藏','分享']
    list_tl = head_100_video_df.sort_values(by='播放', ascending=False).head(30)['标题'].tolist()

    tl = Timeline()
    for i in list_tl:
        funnel = (
            Funnel(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add(
                "作品数据",
                [list(z) for z in zip(list_funnel, np.array(head_100_video_df[head_100_video_df['标题']==i][list_funnel]).flatten().tolist())],
                label_opts=opts.LabelOpts(position="inside")
            )
            .set_global_opts(title_opts=opts.TitleOpts(title=""))
        )
        tl.add(funnel, time_point = i)
    st_pyecharts(tl,height="400px",width="1000px")

    st.write('---')
    st.write("""
    ##### top20弹幕-评论量趋势图
    """)
   
    bar = Bar(init_opts=opts.InitOpts(theme='dark',
                                    width='1000px',
                                    height='600px',)
                                    )
    bar.add_xaxis(head_100_video_df['标题'].tolist()[:30])
    # 添加一个Y轴
    bar.extend_axis(yaxis=opts.AxisOpts(type_="value",
                                        position="right",
                                        is_scale=True,
                                        axislabel_opts=opts.LabelOpts(margin=20, color="#74673E",
                                                                    formatter=
                                                                    JsCode(
                                                                """function (value)
                                                                {return Math.floor(value);}""")),
                                        axisline_opts=opts.AxisLineOpts(
                                            linestyle_opts=opts.LineStyleOpts(
                                                width=2, color="#B5CAA0")
                                        ),
                                        axistick_opts=opts.AxisTickOpts(
                                            is_show=True,
                                            length=15,
                                            linestyle_opts=opts.LineStyleOpts(
                                                color="#D9CD90")
                                        ),
                                        ))
    bar.add_yaxis('弹幕量', head_100_video_df['弹幕'].tolist()[:30], yaxis_index=0,
    #               z_level=0,
                category_gap='30%',
                itemstyle_opts=opts.ItemStyleOpts(color='#00AA90', 
                                                    opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(
                        # visualmap_opts=opts.VisualMapOpts(type_='color', min_=500, max_=2000,series_index=0,
                        #                                   range_color=['#0071ce', '#ffc220', '#ffffff']),
                        title_opts=opts.TitleOpts(title="",
                                                pos_left="center",
                                                pos_top='1%',
                                                title_textstyle_opts=opts.TextStyleOpts(
                                                    font_size=20,
                                                    color='#00BFFF')),
                        legend_opts=opts.LegendOpts(is_show=True, pos_top='6%'),
                        xaxis_opts=opts.AxisOpts(boundary_gap=False,
                                                is_show = False,
                                                axislabel_opts=opts.LabelOpts(
                                                    margin=30, color="#74673E"),
                                                axisline_opts=opts.AxisLineOpts(
                                                    is_show=False),
                                                axistick_opts=opts.AxisTickOpts(
                                                    is_show=True,
                                                    length=10,
                                                    linestyle_opts=opts.LineStyleOpts(
                                                        color="#D9CD90"),
                                                ),
                                                splitline_opts=opts.SplitLineOpts(
                                                    is_show=True, linestyle_opts=opts.LineStyleOpts(
                                                        color="#D9CD90")
                                                ),
                                                ),
                        yaxis_opts=opts.AxisOpts(
        type_="value",
        position="left",
        is_scale=True,
        axislabel_opts=opts.LabelOpts(margin=20, 
                                    color="#74673E",
                                    formatter=JsCode(
                                        """function (value) {return Math.floor(value);}""")),
        axisline_opts=opts.AxisLineOpts(
            linestyle_opts=opts.LineStyleOpts(
                width=2, color="#B5CAA0")
        ),
        axistick_opts=opts.AxisTickOpts(
            is_show=True,
            length=15,
            linestyle_opts=opts.LineStyleOpts(
                color="#D9CD90"),
        ),
        splitline_opts=opts.SplitLineOpts(
            is_show=True, linestyle_opts=opts.LineStyleOpts(
                color="#D9CD90")
        ),
    )
    )

    line = Line(init_opts=opts.InitOpts(theme='light',
                                        width='1000px',
                                        height='600px'))
    line.add_xaxis(head_100_video_df['标题'][:30].tolist(),
                )
    # 将line数据通过yaxis_index指向后添加的Y轴
    line.add_yaxis('评论数', head_100_video_df['评论'][:30].tolist(), yaxis_index=1,
                is_smooth=True,
                symbol_size=8,
                color='red',
                z_level=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color='#563F2E'),
                linestyle_opts={
                    'normal': {
                        'width': 3,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)',
                        'shadowBlur': 5,
                        'shadowOffsetY': 10,
                        'shadowOffsetX': 10,
                        'curve': 0.5,
                        'color': '#E16B8C'
                    }
                })

    bar.overlap(line)
    st_pyecharts(bar,width=1000,height=600)

    st.write('---')
    st.write("""
    ###  热门视频的时长分布
    """)

    st.write('---')

    df_duration = pd.read_csv('data/bilibili_data_analysis_duration.csv')
    df_duration = df_duration.drop_duplicates(subset=['标题'],keep='last',inplace=False)#按标题去重
    #缺失值处理
    df_duration = df_duration.drop(df_duration[df_duration['时长']>10000].index)
    df_duration[df_duration.isnull().values==True]
    #重置索引
    df_duration=df_duration.reset_index(drop=True, inplace=False)

    df_duration['时长'] = df_duration['时长'].values/60

    #统计各个时间段视频数量
    k= 0
    video_number_list = []
    xlable_time = [] #用作横坐标参数
    for i in range(0,31):
        k=i
        day_to_day = str(k)+'-'+str(k+1)+'分钟'
        xlable_time.append(day_to_day)
        video_number_list.append(len(df_duration[(df_duration['时长'].values<k+1) & (df_duration['时长'].values>=k)]))
    video_number_list.append(len(df_duration[df_duration['时长'].values>=31]))
    xlable_time.append('>=31')

    b6 =(
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK,height='500px',width='1000px'))
        .add_xaxis(xlable_time)
        .add_yaxis("视频数量",video_number_list,label_opts=opts.LabelOpts(is_show=False,position='top'), itemstyle_opts=opts.ItemStyleOpts(
                                        color=JsCode("""new echarts.graphic.LinearGradient(0, 0, 0, 1, 
                                                    [{
                                                        offset: 0,
                                                        color: 'rgb(255,99,71)'
                                                    }, {
                                                        offset: 1,
                                                        color: 'rgb(32,178,170)'
                                                    }])"""))
                        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=""),
            toolbox_opts=opts.ToolboxOpts(),
            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(trigger='axis',axis_pointer_type='cross'),
            xaxis_opts=opts.AxisOpts(name='视频时长',
                                                    type_='category',                                           
                                                    axislabel_opts=opts.LabelOpts(rotate=45),
                                                    ),
            yaxis_opts=opts.AxisOpts(name='', splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(type_='dash')),

                                    )
        )  
        .render_embed()
    ) 
    components.html(b6, width=1000, height=500)

    
