import streamlit as st

from Project_Introduction import introduction
from Data_Analysis import data_analysis
from Model_show import *
from Regression_Prediction import *

def main():
    PAGES = {
        "集成学习": Model_show,
        "随机森林": Random_Forest,
        "GBDT": GBDT,
        "XGBoost": XGBoost,
        "LightGBM": LightGBM,

    }

    st.sidebar.header("1. 设置")

    api_options = ("项目介绍", "数据集分析","模型展示","回归预测")
    select_api = st.sidebar.selectbox(
        label="功能选择:", options=api_options,
    )

    if select_api == "项目介绍":
        introduction()
    if select_api == "数据集分析":
        data_analysis()
    if select_api == "模型展示":
        page = st.sidebar.selectbox("模型选择", options=list(PAGES.keys()))
        PAGES[page]()
    if select_api == "回归预测":
        Regression_Prediction()
    



def set_global():
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
    else:
        st.session_state.first_visit = False
    # 初始化全局配置
    if st.session_state.first_visit:
       st.balloons()


if __name__ == "__main__":
    st.set_page_config(
        page_title="B站视频数据分析预测",
        layout="wide",
        page_icon="https://www.bilibili.com/favicon.ico",
        initial_sidebar_state="expanded",
    )
    set_global()
    
    main()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp&nbsp <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp&nbsp by &nbsp&nbsp&nbsp<a href="https://github.com/RubyRose-TAT/demo" style="font-family: 宋体; font-size:126%;">OwO</a></h6>',
            unsafe_allow_html=True
        )
