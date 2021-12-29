import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import plotly.graph_objects as go

import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import streamlit.components.v1 as components


def Model_show():

    #---------------------------------#
    st.write("""
    # 集成学习
    **随机森林** 超参数调整演示

    示例使用随机森林算法，通过 *RandomForestRegressor()* 函数，建立回归模型

    """)

    with st.expander("集成学习介绍"):
        st.write("""
                    ---
                    *集成学习*（ensemble learning）通过构建并组合多个学习器来完成学习任务

                    **如何产生并结合好而不同的个体学习器，恰是集成学习研究的核心**

                    集成学习的思路是通过合并多个模型来提升机器学习性能，将多个基学习器结合，通常都会获得比单一学习器更显著优越的泛化性能，相较于当个单个模型能够获得更好的预测结果

                    ---

                    集成学习在各个规模的数据集上都有很好的策略
                    - 数据集小：利用Bootstrap方法进行抽样，得到多个数据集，分别训练多个模型再进行组合
                    - 数据集大：划分成多个小数据集，学习多个模型进行组合

                    ---

                    一般来说集成学习可以分为三大类：
                    - 用于减少方差的bagging
                    - 用于减少偏差的boosting
                    - 用于提升预测结果的stacking
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


def Random_Forest():
    st.write("""
    # 随机森林
    """)
    st.markdown("---")

    with st.expander("随机森林介绍"):
        st.write("""
            ## 1.什么是随机森林

            ### 1.1 Bagging思想

            Bagging是bootstrap aggregating。思想就是从总体样本当中随机取一部分样本进行训练，通过多次这样的结果，进行投票获取平均值作为结果输出，这就极大可能的避免了不好的样本数据，从而提高准确度。因为有些是不好的样本，相当于噪声，模型学入噪声后会使准确度不高。

            **举个例子**：

            假设有1000个样本，如果按照以前的思维，是直接把这1000个样本拿来训练，但现在不一样，先抽取800个样本来进行训练，假如噪声点是这800个样本以外的样本点，就很有效的避开了。重复以上操作，提高模型输出的平均值。

            ### 1.2 随机森林

            Random Forest(随机森林)是一种基于树模型的Bagging的优化版本，一棵树的生成肯定还是不如多棵树，因此就有了随机森林，解决决策树泛化能力弱的特点。

            而同一批数据，用同样的算法只能产生一棵树，这时Bagging策略可以帮助我们产生不同的数据集。**Bagging**策略来源于bootstrap aggregation：从样本集（假设样本集N个数据点）中重采样选出Nb个样本（有放回的采样，样本数据点个数仍然不变为N），在所有样本上，对这n个样本建立分类器（ID3\C4.5\CART\SVM\LOGISTIC），重复以上两步m次，获得m个分类器，最后根据这m个分类器的投票结果，决定数据属于哪一类。

            **每棵树的按照如下规则生成：**

            1. 如果训练集大小为N，对于每棵树而言，**随机**且有放回地从训练集中的抽取N个训练样本，作为该树的训练集；
            2. 如果每个样本的特征维度为M，指定一个常数m<<M，**随机**地从M个特征中选取m个特征子集，每次树进行分裂时，从这m个特征中选择最优的；
            3. 每棵树都尽最大程度的生长，并且没有剪枝过程。

            一开始我们提到的随机森林中的“随机”就是指的这里的两个随机性。两个随机性的引入对随机森林的分类性能至关重要。由于它们的引入，使得随机森林不容易陷入过拟合，并且具有很好得抗噪能力（比如：对缺省值不敏感）。

            总的来说就是随机选择样本数，随机选取特征，随机选择分类器，建立多颗这样的决策树，然后通过这几课决策树来投票，决定数据属于哪一类(**投票机制有一票否决制、少数服从多数、加权多数**)

            ## 2. 随机森林分类效果的影响因素

            - 森林中任意两棵树的相关性：相关性越大，错误率越大；
            - 森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。

            减小特征选择个数m，树的相关性和分类能力也会相应的降低；增大m，两者也会随之增大。所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。

            ## 3. 随机森林有什么优缺点

            **优点：**

            - 在当前的很多数据集上，相对其他算法有着很大的优势，表现良好。
            - 它能够处理很高维度（feature很多）的数据，并且不用做特征选择(因为特征子集是随机选择的)。
            - 在训练完后，它能够给出哪些feature比较重要。
            - 训练速度快，容易做成并行化方法(训练时树与树之间是相互独立的)。
            - 在训练过程中，能够检测到feature间的互相影响。
            - 对于不平衡的数据集来说，它可以平衡误差。
            - 如果有很大一部分的特征遗失，仍可以维持准确度。

            **缺点：**

            - 随机森林已经被证明在某些**噪音较大**的分类或回归问题上会过拟合。
            - 对于有不同取值的属性的数据，取值划分较多的属性会对随机森林产生更大的影响，所以随机森林在这种数据上产出的属性权值是不可信的。

            ## 4. 随机森林如何处理缺失值？

            根据随机森林创建和训练的特点，随机森林对缺失值的处理还是比较特殊的。

            - 首先，给缺失值预设一些估计值，比如数值型特征，选择其余数据的中位数或众数作为当前的估计值
            - 然后，根据估计的数值，建立随机森林，把所有的数据放进随机森林里面跑一遍。记录每一组数据在决策树中一步一步分类的路径.
            - 判断哪组数据和缺失数据路径最相似，引入一个相似度矩阵，来记录数据之间的相似度，比如有N组数据，相似度矩阵大小就是N*N
            - 如果缺失值是类别变量，通过权重投票得到新估计值，如果是数值型变量，通过加权平均得到新的估计值，如此迭代，直到得到稳定的估计值。

            其实，该缺失值填补过程类似于推荐系统中采用协同过滤进行评分预测，先计算缺失特征与其他特征的相似度，再加权得到缺失值的估计，而随机森林中计算相似度的方法（数据在决策树中一步一步分类的路径）乃其独特之处。

            ## 5. 什么是OOB？随机森林中OOB是如何计算的，它有什么优缺点？

            **OOB**：

            上面我们提到，构建随机森林的关键问题就是如何选择最优的m，要解决这个问题主要依据计算袋外错误率oob error（out-of-bag error）。

            bagging方法中Bootstrap每次约有1/3的样本不会出现在Bootstrap所采集的样本集合中，当然也就没有参加决策树的建立，把这1/3的数据称为**袋外数据oob（out of bag）**,它可以用于取代测试集误差估计方法。

            **袋外数据(oob)误差的计算方法如下：**

            - 对于已经生成的随机森林,用袋外数据测试其性能,假设袋外数据总数为O,用这O个袋外数据作为输入,带进之前已经生成的随机森林分类器,分类器会给出O个数据相应的分类
            - 因为这O条数据的类型是已知的,则用正确的分类与随机森林分类器的结果进行比较,统计随机森林分类器分类错误的数目,设为X,则袋外数据误差大小=X/O

            **优缺点**：

            这已经经过证明是无偏估计的,所以在随机森林算法中不需要再进行交叉验证或者单独的测试集来获取测试集误差的无偏估计。 

            ## 6. 随机森林的过拟合问题

            1. 你已经建了一个有10000棵树的随机森林模型。在得到0.00的训练误差后，你非常高兴。但是，验证错误是34.23。到底是怎么回事？你还没有训练好你的模型吗？

            答：该模型过度拟合，因此，为了避免这些情况，我们要用交叉验证来调整树的数量。
        """)
        
    st.subheader('1. 数据集')

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
    X = df.drop(["views"],axis = 1)
    y = df["views"]

    st.markdown('bilibili排行榜RANK100数据集')
    st.write(df.head(10))

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)

    st.markdown('**1.2. 数据划分**')
    st.write('训练集')
    st.info(X_train.shape)
    st.write('测试集')
    st.info(X_test.shape)

    st.markdown('**1.3. 变量细节**:')
    st.write('变量X')
    st.info(list(X.columns))
    st.write('变量Y')
    st.info(y.name)
        
    load_model = pickle.load(open('models/rfFinal.pkl', 'rb'))

    st.subheader('2. 模型性能')

    st.markdown('**测试集**')
    Y_pred_test = load_model.predict(X_test)
    st.write('决定系数 ($R^2$):')
    st.info( r2_score(y_test, Y_pred_test) )
    st.write('均方误差 (RMSE):')
    st.info( mean_squared_error(y_test, Y_pred_test) )
    st.write('均方根误差 (RMSE):')
    st.info( np.sqrt(mean_squared_error(y_test, Y_pred_test)) )

    st.sidebar.subheader('2.随机森林模型超参数')
    st.sidebar.write(load_model.get_params())

    st.write("""
    ##### 代码实现
    GitHub：[https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/Random_Forest.ipynb](https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/2.bilibili_%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97.ipynb)
    """)


def GBDT():
    st.write("""
    # GBDT(梯度提升决策树)
    """)
    st.markdown("---")

    with st.expander("GBDT介绍"):
        st.write("""
            ## 1. 解释一下GBDT算法的过程

            GBDT(Gradient Boosting Decision Tree)，全名叫梯度提升决策树，使用的是**Boosting**的思想。

            ### 1.1 Boosting思想

            Boosting方法训练基分类器时采用串行的方式，各个基分类器之间有依赖。它的基本思路是将基分类器层层叠加，每一层在训练的时候，对前一层基分类器分错的样本，给予更高的权重。测试时，根据各层分类器的结果的加权得到最终结果。 

            Bagging与Boosting的串行训练方式不同，Bagging方法在训练过程中，各基分类器之间无强依赖，可以进行并行训练。

            ### 1.2 GBDT原来是这么回事

            GBDT的原理很简单，就是所有弱分类器的结果相加等于预测值，然后下一个弱分类器去拟合误差函数对预测值的残差(这个残差就是预测值与真实值之间的误差)。当然了，它里面的弱分类器的表现形式就是各棵树。

            举一个非常简单的例子，比如我今年30岁了，但计算机或者模型GBDT并不知道我今年多少岁，那GBDT咋办呢？

            - 它会在第一个弱分类器（或第一棵树中）随便用一个年龄比如20岁来拟合，然后发现误差有10岁；
            - 接下来在第二棵树中，用6岁去拟合剩下的损失，发现差距还有4岁；
            - 接着在第三棵树中用3岁拟合剩下的差距，发现差距只有1岁了；
            - 最后在第四课树中用1岁拟合剩下的残差，完美。
            - 最终，四棵树的结论加起来，就是真实年龄30岁（实际工程中，gbdt是计算负梯度，用负梯度近似残差）。

            **为何gbdt可以用用负梯度近似残差呢？**

            回归任务下，GBDT 在每一轮的迭代时对每个样本都会有一个预测值，此时的损失函数为均方差损失函数，

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214962034944638.gif)

            那此时的负梯度是这样计算的

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214962416670973.gif)

            所以，当损失函数选用均方损失函数是时，每一次拟合的值就是（真实值 - 当前模型预测的值），即残差。此时的变量是![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155214963633267938.gif)，即“当前预测模型的值”，也就是对它求负梯度。

            **训练过程**

            简单起见，假定训练集只有4个人：A,B,C,D，他们的年龄分别是14,16,24,26。其中A、B分别是高一和高三学生；C,D分别是应届毕业生和工作两年的员工。如果是用一棵传统的回归决策树来训练，会得到如下图所示结果：

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438568191303958.png)

            现在我们使用GBDT来做这件事，由于数据太少，我们限定叶子节点做多有两个，即每棵树都只有一个分枝，并且限定只学两棵树。我们会得到如下图所示结果：

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438570529256895.png)

            在第一棵树分枝和图1一样，由于A,B年龄较为相近，C,D年龄较为相近，他们被分为左右两拨，每拨用平均年龄作为预测值。

            - 此时计算残差（残差的意思就是：A的实际值 - A的预测值 = A的残差），所以A的残差就是实际值14 - 预测值15 = 残差值-1。
            - 注意，A的预测值是指前面所有树累加的和，这里前面只有一棵树所以直接是15，如果还有树则需要都累加起来作为A的预测值。

            然后拿它们的残差-1、1、-1、1代替A B C D的原值，到第二棵树去学习，第二棵树只有两个值1和-1，直接分成两个节点，即A和C分在左边，B和D分在右边，经过计算（比如A，实际值-1 - 预测值-1 = 残差0，比如C，实际值-1 - 预测值-1 = 0），此时所有人的残差都是0。残差值都为0，相当于第二棵树的预测值和它们的实际值相等，则只需把第二棵树的结论累加到第一棵树上就能得到真实年龄了，即每个人都得到了真实的预测值。

            换句话说，现在A,B,C,D的预测值都和真实年龄一致了。Perfect！

            - A: 14岁高一学生，购物较少，经常问学长问题，预测年龄A = 15 – 1 = 14
            - B: 16岁高三学生，购物较少，经常被学弟问问题，预测年龄B = 15 + 1 = 16
            - C: 24岁应届毕业生，购物较多，经常问师兄问题，预测年龄C = 25 – 1 = 24
            - D: 26岁工作两年员工，购物较多，经常被师弟问问题，预测年龄D = 25 + 1 = 26

            所以，GBDT需要将多棵树的得分累加得到最终的预测得分，且每一次迭代，都在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差。

            ## 2. 梯度提升和梯度下降的区别和联系是什么？ 

            下表是梯度提升算法和梯度下降算法的对比情况。可以发现，两者都是在每 一轮迭代中，利用损失函数相对于模型的负梯度方向的信息来对当前模型进行更 新，只不过在梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参 数的更新。而在梯度提升中，模型并不需要进行参数化表示，而是直接定义在函 数空间中，从而大大扩展了可以使用的模型种类。

            ![](http://wx3.sinaimg.cn/mw690/00630Defgy1g4tdwhqzsdj30rp0afdho.jpg)

            ## 3. **GBDT**的优点和局限性有哪些？ 

            ### 3.1 优点

            1. 预测阶段的计算速度快，树与树之间可并行化计算。
            2. 在分布稠密的数据集上，泛化能力和表达能力都很好，这使得GBDT在Kaggle的众多竞赛中，经常名列榜首。 
            3. 采用决策树作为弱分类器使得GBDT模型具有较好的解释性和鲁棒性，能够自动发现特征间的高阶关系，并且也不需要对数据进行特殊的预处理如归一化等。

            ### 3.2 局限性

            1. GBDT在高维稀疏的数据集上，表现不如支持向量机或者神经网络。
            2. GBDT在处理文本分类特征问题上，相对其他模型的优势不如它在处理数值特征时明显。 
            3. 训练过程需要串行训练，只能在决策树内部采用一些局部并行的手段提高训练速度。 

            ## 4. RF(随机森林)与GBDT之间的区别与联系

            **相同点**：

            都是由多棵树组成，最终的结果都是由多棵树一起决定。

            **不同点**：

            - 组成随机森林的树可以分类树也可以是回归树，而GBDT只由回归树组成
            - 组成随机森林的树可以并行生成，而GBDT是串行生成
            - 随机森林的结果是多数表决表决的，而GBDT则是多棵树累加之和
            - 随机森林对异常值不敏感，而GBDT对异常值比较敏感
            - 随机森林是减少模型的方差，而GBDT是减少模型的偏差
            - 随机森林不需要进行特征归一化。而GBDT则需要进行特征归一化
        """)
        
    st.subheader('1. 数据集')

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
    X = df.drop(["views"],axis = 1)
    y = df["views"]

    st.markdown('bilibili排行榜RANK100数据集')
    st.write(df.head(10))

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)

    st.markdown('**1.2. 数据划分**')
    st.write('训练集')
    st.info(X_train.shape)
    st.write('测试集')
    st.info(X_test.shape)

    st.markdown('**1.3. 变量细节**:')
    st.write('变量X')
    st.info(list(X.columns))
    st.write('变量Y')
    st.info(y.name)
        
    load_model = pickle.load(open('models/gbdtFinal.pkl', 'rb'))

    st.subheader('2. 模型性能')

    st.markdown('**测试集**')
    Y_pred_test = load_model.predict(X_test)
    st.write('决定系数 ($R^2$):')
    st.info( r2_score(y_test, Y_pred_test) )
    st.write('均方误差 (RMSE):')
    st.info( mean_squared_error(y_test, Y_pred_test) )
    st.write('均方根误差 (RMSE):')
    st.info( np.sqrt(mean_squared_error(y_test, Y_pred_test)) )

    st.sidebar.subheader('2.GBDT模型超参数')
    st.sidebar.write(load_model.get_params())

    st.write("""
    ##### 代码实现
    GitHub：[https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/3.bilibili_GBDT.ipynb](https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/3.bilibili_GBDT.ipynb)
    """)


def XGBoost():
    st.write("""
    # XGBoost
    """)
    st.markdown("---")

    with st.expander("XGBoost介绍"):
        st.write("""
            ## 1. 什么是XGBoost

            XGBoost是陈天奇等人开发的一个开源机器学习项目，高效地实现了GBDT算法并进行了算法和工程上的许多改进，被广泛应用在Kaggle竞赛及其他许多机器学习竞赛中并取得了不错的成绩。

            说到XGBoost，不得不提GBDT(Gradient Boosting Decision Tree)。因为XGBoost本质上还是一个GBDT，但是力争把速度和效率发挥到极致，所以叫X (Extreme) GBoosted。包括前面说过，两者都是boosting方法。

            ### 1.1 XGBoost树的定义

            先来举个**例子**，我们要预测一家人对电子游戏的喜好程度，考虑到年轻和年老相比，年轻更可能喜欢电子游戏，以及男性和女性相比，男性更喜欢电子游戏，故先根据年龄大小区分小孩和大人，然后再通过性别区分开是男是女，逐一给各人在电子游戏喜好程度上打分，如下图所示。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438577232516800.png)

            就这样，训练出了2棵树tree1和tree2，类似之前gbdt的原理，两棵树的结论累加起来便是最终的结论，所以小孩的预测分数就是两棵树中小孩所落到的结点的分数相加：2 + 0.9 = 2.9。爷爷的预测分数同理：-1 + （-0.9）= -1.9。具体如下图所示：

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438578739198433.png)

            恩，你可能要拍案而起了，惊呼，这不是跟上文介绍的GBDT乃异曲同工么？

            事实上，如果不考虑工程实现、解决问题上的一些差异，XGBoost与GBDT比较大的不同就是目标函数的定义。XGBoost的目标函数如下图所示：

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438580139159593.png)

            其中：

            - 红色箭头所指向的L 即为损失函数（比如平方损失函数：![](https://latex.codecogs.com/gif.latex?l(y_i,y^i)=(y_i-y^i)^2))
            - 红色方框所框起来的是正则项（包括L1正则、L2正则）
            - 红色圆圈所圈起来的为常数项
            - 对于f(x)，XGBoost利用泰勒展开三项，做一个近似。**f(x)表示的是其中一颗回归树。**

            看到这里可能有些读者会头晕了，这么多公式，**我在这里只做一个简要式的讲解，具体的算法细节和公式求解请查看这篇博文，讲得很仔细**：[通俗理解kaggle比赛大杀器xgboost](https://blog.csdn.net/v_JULY_v/article/details/81410574)

            XGBoost的**核心算法思想**不难，基本就是：

            1. 不断地添加树，不断地进行特征分裂来生长一棵树，每次添加一个树，其实是学习一个新函数**f(x)**，去拟合上次预测的残差。
            2. 当我们训练完成得到k棵树，我们要预测一个样本的分数，其实就是根据这个样本的特征，在每棵树中会落到对应的一个叶子节点，每个叶子节点就对应一个分数
            3. 最后只需要将每棵树对应的分数加起来就是该样本的预测值。

            显然，我们的目标是要使得树群的预测值![](https://latex.codecogs.com/gif.latex?y_i^{'})尽量接近真实值![](https://latex.codecogs.com/gif.latex?y_i)，而且有尽量大的泛化能力。类似之前GBDT的套路，XGBoost也是需要将多棵树的得分累加得到最终的预测得分（每一次迭代，都在现有树的基础上，增加一棵树去拟合前面树的预测结果与真实值之间的残差）。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438657261833493.png)

            那接下来，我们如何选择每一轮加入什么 f 呢？答案是非常直接的，选取一个 f 来使得我们的目标函数尽量最大地降低。这里 f 可以使用泰勒展开公式近似。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase6415343865867530120.png)

            实质是把样本分配到叶子结点会对应一个obj，优化过程就是obj优化。也就是分裂节点到叶子不同的组合，不同的组合对应不同obj，所有的优化围绕这个思想展开。到目前为止我们讨论了目标函数中的第一个部分：训练误差。接下来我们讨论目标函数的第二个部分：正则项，即如何定义树的复杂度。

            ### 1.2 正则项：树的复杂度

            XGBoost对树的复杂度包含了两个部分：

            - 一个是树里面叶子节点的个数T
            - 一个是树上叶子节点的得分w的L2模平方（对w进行L2正则化，相当于针对每个叶结点的得分增加L2平滑，目的是为了避免过拟合）

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64153438674199471483.png)

            我们再来看一下XGBoost的目标函数（损失函数揭示训练误差 + 正则化定义复杂度）：

            ![](https://latex.codecogs.com/gif.latex?L(\phi)=\sum_{i}l(y_i^{'}-y_i)+\sum_k\Omega(f_t))

            正则化公式也就是目标函数的后半部分，对于上式而言，![](https://latex.codecogs.com/gif.latex?y_i^{'})是整个累加模型的输出，正则化项∑kΩ(ft)是则表示树的复杂度的函数，值越小复杂度越低，泛化能力越强。

            ### 1.3 树该怎么长

            很有意思的一个事是，我们从头到尾了解了xgboost如何优化、如何计算，但树到底长啥样，我们却一直没看到。很显然，一棵树的生成是由一个节点一分为二，然后不断分裂最终形成为整棵树。那么树怎么分裂的就成为了接下来我们要探讨的关键。对于一个叶子节点如何进行分裂，XGBoost作者在其原始论文中给出了一种分裂节点的方法：**枚举所有不同树结构的贪心法**

            不断地枚举不同树的结构，然后利用打分函数来寻找出一个最优结构的树，接着加入到模型中，不断重复这样的操作。这个寻找的过程使用的就是**贪心算法**。选择一个feature分裂，计算loss function最小值，然后再选一个feature分裂，又得到一个loss function最小值，你枚举完，找一个效果最好的，把树给分裂，就得到了小树苗。

            总而言之，XGBoost使用了和CART回归树一样的想法，利用贪婪算法，遍历所有特征的所有特征划分点，不同的是使用的目标函数不一样。具体做法就是分裂后的目标函数值比单子叶子节点的目标函数的增益，同时为了限制树生长过深，还加了个阈值，只有当增益大于该阈值才进行分裂。从而继续分裂，形成一棵树，再形成一棵树，**每次在上一次的预测基础上取最优进一步分裂/建树。**

            ### 1.4 如何停止树的循环生成

            凡是这种循环迭代的方式必定有停止条件，什么时候停止呢？简言之，设置树的最大深度、当样本权重和小于设定阈值时停止生长以防止过拟合。具体而言，则

            1. 当引入的分裂带来的增益小于设定阀值的时候，我们可以忽略掉这个分裂，所以并不是每一次分裂loss function整体都会增加的，有点预剪枝的意思，阈值参数为（即正则项里叶子节点数T的系数）；
            2. 当树达到最大深度时则停止建立决策树，设置一个超参数max_depth，避免树太深导致学习局部样本，从而过拟合；
            3. 样本权重和小于设定阈值时则停止建树。什么意思呢，即涉及到一个超参数-最小的样本权重和min_child_weight，和GBM的 min_child_leaf 参数类似，但不完全一样。大意就是一个叶子节点样本太少了，也终止同样是防止过拟合；

            ## 2. XGBoost与GBDT有什么不同

            除了算法上与传统的GBDT有一些不同外，XGBoost还在工程实现上做了大量的优化。总的来说，两者之间的区别和联系可以总结成以下几个方面。 

            1. GBDT是机器学习算法，XGBoost是该算法的工程实现。
            2. 在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模 型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。
            3. GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代 价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。
            4. 传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类 器，比如线性分类器。
            5. 传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机 森林相似的策略，支持对数据进行采样。
            6. 传统的GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺 失值的处理策略。

            ## 3. 为什么XGBoost要用泰勒展开，优势在哪里？

            XGBoost使用了一阶和二阶偏导, 二阶导数有利于梯度下降的更快更准. 使用泰勒展开取得函数做自变量的二阶导数形式, 可以在不选定损失函数具体形式的情况下, 仅仅依靠输入数据的值就可以进行叶子分裂优化计算, 本质上也就把损失函数的选取和模型算法优化/参数选择分开了. 这种去耦合增加了XGBoost的适用性, 使得它按需选取损失函数, 可以用于分类, 也可以用于回归。
        """)
        
    st.subheader('1. 数据集')

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
    X = df.drop(["views"],axis = 1)
    y = df["views"]

    st.markdown('bilibili排行榜RANK100数据集')
    st.write(df.head(10))

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)

    st.markdown('**1.2. 数据划分**')
    st.write('训练集')
    st.info(X_train.shape)
    st.write('测试集')
    st.info(X_test.shape)

    st.markdown('**1.3. 变量细节**:')
    st.write('变量X')
    st.info(list(X.columns))
    st.write('变量Y')
    st.info(y.name)
        
    load_model = pickle.load(open('models/xgbFinal.pkl', 'rb'))

    st.subheader('2. 模型性能')

    st.markdown('**测试集**')
    xgbDMat = xgb.DMatrix(data = X_test)
    Y_pred_test = load_model.predict(xgbDMat)
    st.write('决定系数 ($R^2$):')
    st.info( r2_score(y_test, Y_pred_test) )
    st.write('均方误差 (RMSE):')
    st.info( mean_squared_error(y_test, Y_pred_test) )
    st.write('均方根误差 (RMSE):')
    st.info( np.sqrt(mean_squared_error(y_test, Y_pred_test)) )

    st.sidebar.subheader('2.XGBoost模型超参数')
    st.sidebar.write({'n_estimators': 450, 'learning_rate': 0.1, 'gamma': 0.1, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_alpha': 100, 'reg_lambda': 1e-05, 'max_depth': 5, 'min_child_weight': 6})

    st.write("""
    ##### 代码实现
    GitHub：[https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/4.bilibili_XGBoost.ipynb](https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/4.bilibili_XGBoost.ipynb)
    """)


def LightGBM():
    st.write("""
    # LightGBM
    """)
    st.markdown("---")

    with st.expander("LightGBM介绍"):
        st.write("""
            ## 1. 什么是LightGBM

            不久前微软DMTK(分布式机器学习工具包)团队在GitHub上开源了性能超越其他boosting工具的LightGBM，在三天之内GitHub上被star了1000次，fork了200次。知乎上有近千人关注“如何看待微软开源的LightGBM？”问题，被评价为“速度惊人”，“非常有启发”，“支持分布式”，“代码清晰易懂”，“占用内存小”等。

            LightGBM （Light Gradient Boosting Machine）(请点击[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM))是一个实现GBDT算法的框架，支持高效率的并行训练。

            LightGBM在Higgs数据集上LightGBM比XGBoost快将近10倍，内存占用率大约为XGBoost的1/6，并且准确率也有提升。GBDT在每一次迭代的时候，都需要遍历整个训练数据多次。如果把整个训练数据装进内存则会限制训练数据的大小；如果不装进内存，反复地读写训练数据又会消耗非常大的时间。尤其面对工业级海量的数据，普通的GBDT算法是不能满足其需求的。

            LightGBM提出的主要原因就是为了解决GBDT在海量数据遇到的问题，让GBDT可以更好更快地用于工业实践。

            ### 1.1 LightGBM在哪些地方进行了优化    (区别XGBoost)？

            - 基于Histogram的决策树算法
            - 带深度限制的Leaf-wise的叶子生长策略
            - 直方图做差加速直接
            - 支持类别特征(Categorical Feature)
            - Cache命中率优化
            - 基于直方图的稀疏特征优化多线程优化。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197431597512984.jpg)

            ### 1.2 Histogram算法

            直方图算法的基本思想是先把连续的浮点特征值离散化成k个整数（其实又是分桶的思想，而这些桶称为bin，比如[0,0.1)→0, [0.1,0.3)→1），同时构造一个宽度为k的直方图。

            在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197418746568601.jpg)

            使用直方图算法有很多优点。首先，最明显就是内存消耗的降低，直方图算法不仅不需要额外存储预排序的结果，而且可以只保存特征离散化后的值，而这个值一般用8位整型存储就足够了，内存消耗可以降低为原来的1/8。然后在计算上的代价也大幅降低，预排序算法每遍历一个特征值就需要计算一次分裂的增益，而直方图算法只需要计算k次（k可以认为是常数），时间复杂度从O(#data*#feature)优化到O(k*#features)。

            ### 1.3 带深度限制的Leaf-wise的叶子生长策略

            在XGBoost中，树是按层生长的，称为Level-wise tree growth，同一层的所有节点都做分裂，最后剪枝，如下图所示：

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197509149646916.png)

            Level-wise过一次数据可以同时分裂同一层的叶子，容易进行多线程优化，也好控制模型复杂度，不容易过拟合。但实际上Level-wise是一种低效的算法，因为它不加区分的对待同一层的叶子，带来了很多没必要的开销，因为实际上很多叶子的分裂增益较低，没必要进行搜索和分裂。

            在Histogram算法之上，LightGBM进行进一步的优化。首先它抛弃了大多数GBDT工具使用的按层生长 (level-wise)
            的决策树生长策略，而使用了带有深度限制的按叶子生长 (leaf-wise)算法。

            ![](https://julyedu-img.oss-cn-beijing.aliyuncs.com/quesbase64155197520844369289.png)

            Leaf-wise则是一种更为高效的策略，每次从当前所有叶子中，找到分裂增益最大的一个叶子，然后分裂，如此循环。因此同Level-wise相比，在分裂次数相同的情况下，Leaf-wise可以降低更多的误差，得到更好的精度。Leaf-wise的缺点是可能会长出比较深的决策树，产生过拟合。因此LightGBM在Leaf-wise之上增加了一个最大深度的限制，在保证高效率的同时防止过拟合。

            ### 1.4 直方图差加速

            LightGBM另一个优化是Histogram（直方图）做差加速。一个容易观察到的现象：一个叶子的直方图可以由它的父亲节点的直方图与它兄弟的直方图做差得到。通常构造直方图，需要遍历该叶子上的所有数据，但直方图做差仅需遍历直方图的k个桶。

            利用这个方法，LightGBM可以在构造一个叶子的直方图后，可以用非常微小的代价得到它兄弟叶子的直方图，在速度上可以提升一倍。

            ### 1.5 直接支持类别特征

            实际上大多数机器学习工具都无法直接支持类别特征，一般需要把类别特征，转化到多维的0/1特征，降低了空间和时间的效率。而类别特征的使用是在实践中很常用的。基于这个考虑，LightGBM优化了对类别特征的支持，可以直接输入类别特征，不需要额外的0/1展开。并在决策树算法上增加了类别特征的决策规则。在Expo数据集上的实验，相比0/1展开的方法，训练速度可以加速8倍，并且精度一致。据我们所知，LightGBM是第一个直接支持类别特征的GBDT工具。

            ## 2. LightGBM优点

            LightGBM （Light Gradient Boosting Machine）(请点击[https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM))是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有以下优点：

            - 更快的训练速度
            - 更低的内存消耗
            - 更好的准确率
            - 分布式支持，可以快速处理海量数据
        """)
        
    st.subheader('1. 数据集')

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
    X = df.drop(["views"],axis = 1)
    y = df["views"]

    st.markdown('bilibili排行榜RANK100数据集')
    st.write(df.head(10))

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)

    st.markdown('**1.2. 数据划分**')
    st.write('训练集')
    st.info(X_train.shape)
    st.write('测试集')
    st.info(X_test.shape)

    st.markdown('**1.3. 变量细节**:')
    st.write('变量X')
    st.info(list(X.columns))
    st.write('变量Y')
    st.info(y.name)
        
    load_model = pickle.load(open('models/lgbFinal.pkl', 'rb'))

    st.subheader('2. 模型性能')

    st.markdown('**测试集**')
    Y_pred_test = load_model.predict(X_test, num_iteration = load_model.best_iteration)
    st.write('决定系数 ($R^2$):')
    st.info( r2_score(y_test, Y_pred_test) )
    st.write('均方误差 (RMSE):')
    st.info( mean_squared_error(y_test, Y_pred_test) )
    st.write('均方根误差 (RMSE):')
    st.info( np.sqrt(mean_squared_error(y_test, Y_pred_test)) )

    st.sidebar.subheader('2.LightGBM模型超参数')
    st.sidebar.write({'num_leaves': 11, 'learning_rate': 0.25, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 100, 'reg_lambda': 0.1, 'max_depth': 4, 'min_data_in_leaf': 7, 'feature_pre_filter': 'false'})

    st.write("""
    ##### 代码实现
    GitHub：[https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/5.bilibili_LIghtGBM.ipynb](https://github.com/RubyRose-TAT/bilibili_Data_analysis-Clout_prediction/blob/main/bilibili/5.bilibili_LIghtGBM.ipynb)
    """)