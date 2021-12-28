import shap
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def Regression_Prediction():
    st.write("""
    # 机器学习模型**播放量**预测
    """)
    st.write('---')

    api_options = ("随机森林", "GBDT","XGBoost","LightGBM")
    select_api = st.selectbox(
        label="模型选择:", options=api_options,
    )

    # 导入数据
    df = pd.read_csv('data/bilibili_rank100_data.csv')

    # 数据预处理
    df = df.drop_duplicates(subset=['title'],keep='first',inplace=False)
    df = df.drop(df[df['time']>1000].index)
    dd = df[df.isnull().values==True]
    df = df.reset_index(drop=True, inplace=False)
    def LabelEncoding(df):
        x, dfc = 'partition', df
        key = dfc[x].unique()
        value = [i for i in range(len(key))] 
        Dict = dict(zip(key, value))
        for i in range(len(key)):
            for j in range(dfc.shape[0]):
                if key[i] == dfc[x][j]:
                    dfc[x][j] = Dict[key[i]]
        dfc[x] = dfc[x].astype(np.int64)
        return dfc
    df = LabelEncoding(df)

    df = df.drop(["author","title","tag"],axis = 1)
    df
    X = df.drop(["views"],axis = 1)
    Y = df["views"]
    df

    
    # 加载模型
    if select_api == "随机森林":
        load_model = pickle.load(open('models/rfFinal.pkl', 'rb'))
    if select_api == "GBDT":
        load_model = pickle.load(open('models/gbdtFinal.pkl', 'rb'))
    if select_api == "XGBoost":
        load_model = pickle.load(open('models/xgbFinal.pkl', 'rb'))
    if select_api == "LightGBM":
        load_model = pickle.load(open('models/lgbFinal.pkl', 'rb'))

    # 侧边栏
    st.sidebar.header('2. 请输入参数')

    def user_input_features():
        like = st.sidebar.slider('like', 0, 1000000, 2000, 100)
        coins = st.sidebar.slider('coins', 0, 1000000, 500, 100)
        collect = st.sidebar.slider('collect', 0, 500000, 2000, 10)
        share = st.sidebar.slider('share', 0, 100000, 1000, 10)
        danmu = st.sidebar.slider('danmu', 0, 200000, 1000, 1000)
        reply = st.sidebar.slider('reply', 0, 50000, 200, 10)
        funs = st.sidebar.slider('funs', 0, 10000000, 10000, 1000)
        partition = st.sidebar.slider('partition', X.partition.min(), X.partition.max(), 0)
        time = st.sidebar.slider('time', 0, 600, 10)
        like_rate = st.sidebar.slider('like_rate', float(0), float(1), 0.10, 0.01)
        data = {'partition': partition,
                'funs': funs,
                'like': like,
                'coins': coins,
                'collect': collect,
                'share': share,
                'danmu': danmu,
                'reply': reply,
                'time': time,
                'like_rate': like_rate}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()
    df0 = df.drop(["partition"],axis = 1)
    df0['partition'] = df['partition'] 

    # 主面板

    # 打印设置的参数
    st.header('输入的参数')
    st.write(df0)
    st.write('---')

    # 应用模型进行预测
    if select_api != "XGBoost":
        prediction = load_model.predict(df)
    if select_api == "XGBoost":
        xgbDMat = xgb.DMatrix(data = df)
        prediction = load_model.predict(xgbDMat)

    st.header('播放量预测结果')
    st.info(int(prediction))
    st.write('---')

    st.header('特征重要度')

    explainer = shap.TreeExplainer(load_model)
    shap_values = explainer.shap_values(X)

    fig1 = plt.subplots()
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(fig1, bbox_inches='tight')

    st.write('---')

    fig2 = plt.subplots()
    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(fig2, bbox_inches='tight')