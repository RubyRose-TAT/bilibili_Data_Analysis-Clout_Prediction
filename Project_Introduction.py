import streamlit as st
from PIL import Image

def introduction():
    st.markdown("""
    # BiliBili视频数据分析预测
    """)
    st.markdown('---')
    st.image(Image.open('data/backgrand.png'), caption='')

    st.sidebar.markdown('---')
    st.sidebar.image(Image.open('data/bilibili.jpeg'))