import streamlit as st
from PIL import Image

def introduction():
    st.sidebar.markdown('---')
    st.sidebar.image(Image.open('data/bilibili.jpg'))

    st.markdown("""
    # BiliBili视频数据分析预测
    #### 项目地址: [github](https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction)
    """)

    st.markdown("""
    #### 在线演示: [share.streamlit.io](https://share.streamlit.io/rubyrose-tat/bilibili_data_analysis-clout_prediction/main/main.py)
    """)

    st.markdown('---')

    st.markdown("""
                # 1. 项目背景

                随着互联网的发展，越来越多的人开始从事自媒体的工作。有些创作者靠着一些火热的视频收获了不少，但还有一些创作者的视频热度一直比较低。分析出一些热门视频的“秘密”必将对新人创作者提供一些帮助。

                *本项目将分析B站视频的一些热门视频的播放量与各大因素的关系，随后使用集成学习模型对播放量进行预测*
    """)
    with st.expander(""):
        st.image(Image.open('data/backgrand.png'), caption='')

    st.markdown("""

                # 2. 项目分工

                #### 黄胜鸿：
                1. 编写爬虫程序，数据爬取
                2. 进行数据清洗和特征工程
                3. Adaboost模型搭建与调参
                4. 利用pyechart对数据集可视化分析

                #### 钟珺涛：
                1. 探索性数据分析
                2. 集成学习模型搭建(Random Forest, GBDT, XGBoost, LIghtGBM)
                3. 模型评估
                4. web应用制作

                # 3. 相关技术

                ### 3.1数据获取

                Xpath定位与请求数据接口获取数据

                ### 3.2可视化

                ###### 3.2.1数据可视化

                可视化的过程中，我们使用了python强大的可视化神器——pyechart, 使用它可以绘制出一些好看的图，让数据的呈现更加直观

                ###### 3.2.2可视化平台搭建

                基于streamlit搭建web应用。Streamlit是一个开源的，可以用于快速搭建Web应用的Python库。
                Streamlit官方的定位是服务于机器学习和数据科学的Web应用框架。它能够快速的帮助我们创建定制化的web应用，而且还非常便于和他人分享。只需要将项目保存到github中，项目下需要有requirements.txt文件，就能在https://share.streamlit.io/  上部署。

                ### 3.3运用模型

                ###### 3.3.1 随机森林

                利用多棵树对样本进行训练并预测的一种分类器

                ###### 3.3.2 GBDT

                由多棵决策树组成,所有树的结论累加起来做最终答案

                ###### 3.3.3 XGBoost

                XGBoost高效地实现了GBDT算法并进行了算法和工程上的许多改进

                ###### 3.3.4 LightGBM

                LightGBM是在GBDT算法框架下的一种改进实现，是一种基于决策树算法的快速、分布式、高性能的GBDT框架

                # 4.结论

                ### 对热度的影响因素
    """)

    st.image(Image.open('data/XGBoost_shap.png'), caption='')

    st.markdown("""

                可以看出：对一个视频播放量影响程度：  点赞>分享>收藏>发布时间>硬币

                ### 给新人创作者的建议

                - 针对标题：在标题里可以使用一些表示疑问的词儿，还可以出现与我们标题高频词汇

                - 针对视频时长：控制6分钟以内

                - 针对投稿分区：尽量投稿播放量比较高的分区，比如说

                - 针对内容：重视内容质量提升
    """)
