import streamlit as st
from PIL import Image

def introduction():
    st.sidebar.markdown('---')
    st.sidebar.image(Image.open('data/bilibili.jpeg'))

    st.markdown("""
    # BiliBili视频数据分析预测
    """)

    st.markdown("""
    #### 项目成员 黄胜鸿 钟珺涛
    """)

    st.markdown('---')

    st.markdown("""
                # 1. 项目背景

                随着互联网的发展，越来越多的人开始从事自媒体的工作。有些创作者靠着一些火热的视频收获了不少。但还有一些创作者的视频热度一直比较低。分析出一些热门视频的“秘密”必将对新人创作者提供一些帮助。

                本项目一开始将分析B站视频的一些热门视频的播放量与各大因素的关系，最后通过一些集成学习算法对播放量进行预测
    """)
    with st.expander(""):
        st.image(Image.open('data/backgrand.png'), caption='')

    st.markdown("""

                # 2. 项目分工

                ## 主要工作

                |  成员  | 主要工作                                                     |
                | :----: | :----------------------------------------------------------- |
                | 黄胜鸿 | 1.编写爬虫程序，数据爬取  2.责数据的清洗，特征处理  3.Adaboost模型搭建与调参  4.利用pyechart可视化分析 |
                | 钟珺涛 | 1.探索性数据分析 2.集成学习模型搭建（Random Forest, GBDT, XGBoost, LIghtGBM） 3. 模型评估，参数调整 4.web页面制作 |

                # 3. 相关技术

                ## 3.1数据获取

                Xpath定位与请求数据接口获取数据

                ## 3.2可视化

                #### 3.2.1数据可视化

                可视化的过程中，我们使用了python强大的可视化神器——pyechart, 使用它可以绘制出一些好看的图，让数据的呈现更加直观

                #### 3.2.2可视化平台搭建

                基于streamlit搭建web应用。Streamlit是一个开源的，可以用于快速搭建Web应用的Python库。
                Streamlit官方的定位是服务于机器学习和数据科学的Web应用框架。它能够快速的帮助我们创建定制化的web应用，而且还非常便于和他人分享。只需要将项目保存到github中，项目下需要有requirements.txt文件，就能在https://share.streamlit.io/  上部署。

                #### 3.3运用模型

                ###### 3.3.1 随机森林

                利用多棵树对样本进行训练并预测的一种分类器

                ###### 3.3.2 GBDT

                是一种基于boosting集成学习思想的加法模型，训练时采用前向分布算法进行贪婪的学习，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。

                ###### 3.3.3 XGBoost

                极致梯度提升，是一种基于GBDT的算法或者说工程实现。

                ###### 3.3.4 LightGBM

                LightGBM依然是在GBDT算法框架下的一种改进实现，是一种基于决策树算法的快速、分布式、高性能的GBDT框架，主要说解决的痛点是面对高维度大数据时提高GBDT框架算法的效率和可扩展性

    """)

