import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def Model_show():

    #---------------------------------#
    st.write("""
    # 集成学习超参数调整
    **(回归预测)**

    示例使用随机森林算法，通过 *RandomForestRegressor()* 函数，建立回归模型

    """)

    #---------------------------------#
    # 侧边栏-上传csv文件
    st.sidebar.header('2. 上传CSV格式数据')
    uploaded_file = st.sidebar.file_uploader(
        "上传csv文件", type=["csv"])
    st.sidebar.markdown("""
    [CSV输入文件示例](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    
    """)

    # 侧边栏 - 参数设置
    st.sidebar.header('3. 设置参数')
    split_size = st.sidebar.slider(
        '划分训练集和测试集 (训练集 %)', 10, 90, 80, 5)

    st.sidebar.subheader('3.1. 学习参数')
    parameter_n_estimators = st.sidebar.slider(
        '弱学习器的个数 (n_estimators)', 0, 500, (10, 100), 50)
    parameter_n_estimators_step = st.sidebar.number_input(
        '网格搜索步长 n_estimators', 10)
    st.sidebar.write('---')
    parameter_max_features = st.sidebar.slider(
        '最大特征数 (max_features)', 1, 50, (1, 6), 1)
    st.sidebar.number_input('网格搜索步长 max_features', 1)
    st.sidebar.write('---')
    parameter_min_samples_split = st.sidebar.slider(
        '内部节点再划分所需最小样本数 (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider(
        '叶子节点最少样本数 (min_samples_leaf)', 1, 10, 2, 1)

    st.sidebar.subheader('3.2. 一般参数')
    parameter_random_state = st.sidebar.slider(
        '随机数种子 (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider(
        '性能指标 (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider(
        '构建树时是否使用有放回的随机抽样 (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider(
        '是否采用袋外样本来评估模型的好坏 (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider(
        '并行运行的核心数量 (n_jobs)', options=[1, -1])

    n_estimators_range = np.arange(
        parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    max_features_range = np.arange(
        parameter_max_features[0], parameter_max_features[1]+1, 1)
    
    #构造参数字典，让参数按照根据步长生成的列表序排列组合遍历一遍
    param_grid = dict(max_features=max_features_range,
                      n_estimators=n_estimators_range)

    #---------------------------------#
    # 主页

    # Displays the dataset
    st.subheader('1. 数据集')

    #---------------------------------#
    # Model building

    def filedownload(df):
        csv = df.to_csv(index=False)
        # strings <-> bytes conversions
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="模型性能.csv">下载csv文件</a>'
        return href

    def build_model(df):

        # Data splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

        st.markdown('**1.2. 数据划分**')
        st.write('训练集')
        st.info(X_train.shape)
        st.write('测试集')
        st.info(X_test.shape)

        st.markdown('**1.3. 变量细节**:')
        st.write('变量X')
        st.info(list(X.columns))
        st.write('变量Y')
        st.info(Y.name)

        st.warning("等待模型加载")

        #实例化随机森林回归器
        rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                    random_state=parameter_random_state,
                                    max_features=parameter_max_features,
                                    criterion=parameter_criterion,
                                    min_samples_split=parameter_min_samples_split,
                                    min_samples_leaf=parameter_min_samples_leaf,
                                    bootstrap=parameter_bootstrap,
                                    oob_score=parameter_oob_score,
                                    n_jobs=parameter_n_jobs)
       
        #以随机森林回归器为基础构造网络搜索回归器
        grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        grid.fit(X_train, Y_train)

        st.subheader('2. 模型性能')

        st.markdown('**2.1. 训练集**')
        Y_pred_train = grid.predict(X_train)
        st.write('决定系数 ($R^2$):')
        st.info( r2_score(Y_train, Y_pred_train) )
        st.write('均方根误差 (RMSE):')
        st.info( np.sqrt(mean_squared_error(Y_train, Y_pred_train)) )

        st.markdown('**2.2. 测试集**')
        Y_pred_test = grid.predict(X_test)
        st.write('决定系数 ($R^2$):')
        st.info( r2_score(Y_test, Y_pred_test) )
        st.write('均方根误差 (RMSE):')
        st.info( np.sqrt(mean_squared_error(Y_test, Y_pred_test)) )

        st.write("最佳参数: %s" % grid.best_params_)
        st.write("模型评分: %0.2f" % grid.best_score_)

        st.subheader('3. 模型超参数')
        st.write(grid.get_params())

        #-----Process grid data-----#
        st.markdown('---')
        st.markdown('**三维曲面图**')
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]), pd.DataFrame(
            grid.cv_results_["mean_test_score"], columns=["R2"])], axis=1)
        # Segment data into groups based on the 2 hyperparameters
        grid_contour = grid_results.groupby(
            ['max_features', 'n_estimators']).mean()
        # Pivoting the data
        grid_reset = grid_contour.reset_index()
        grid_reset.columns = ['max_features', 'n_estimators', 'R2']
        grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
        x = grid_pivot.columns.levels[1].values
        y = grid_pivot.index.values
        z = grid_pivot.values

        #-----Plot-----#
        layout = go.Layout(
            xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                    text='n_estimators')
            ),
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                    text='max_features')
            ))
        fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)
        fig.update_layout(title='',
                        scene=dict(
                            xaxis_title='n_estimators',
                            yaxis_title='max_features',
                            zaxis_title='R2'),
                        autosize=False,
                        width=800, height=800,
                        margin=dict(l=65, r=50, b=65, t=90))
        st.plotly_chart(fig)

        #-----Save grid data-----#
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        z = pd.DataFrame(z)
        df = pd.concat([x, y, z], axis=1)
        st.markdown(filedownload(grid_results), unsafe_allow_html=True)

    #---------------------------------#
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

        X = df.iloc[:, :-1]  # Using all column except for the last column as X
        Y = df.iloc[:, -1]  # Selecting the last column as Y

        build_model(df)
    else:
        st.info('请上传CSV文件')
        if st.button('使用示例数据集'):
            df = pd.read_csv('data/bilibili_rank100_data.csv')
            df = df.drop(df[df['time']>1000].index)
            dd = df[df.isnull().values==True]
            df = df.reset_index(drop=True, inplace=False)
            def LabelEncoding(df):
                x, dfc = 'partition', df
                key = dfc[x].unique()  # 将唯一值作为关键字
                value = [i for i in range(len(key))]  # 键值
                Dict = dict(zip(key, value))  # 字典，即键值对
                for i in range(len(key)):
                    for j in range(dfc.shape[0]):
                        if key[i] == dfc[x][j]:
                            dfc[x][j] = Dict[key[i]]
                dfc[x] = dfc[x].astype(np.int64)
                return dfc
            df = LabelEncoding(df)
            df = df.drop(["author","title","tag"],axis = 1)
            X = df.drop(["views"],axis = 1)
            Y = df["views"]

            st.markdown('bilibili排行榜RANK100数据集')
            st.write(df.head(10))

            build_model(df)
