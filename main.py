import streamlit as st
from sklearn.datasets import load_iris, load_digits
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

@st.cache()
def load_data(data_type):
    if data_type == 'iris':
        iris = load_iris()
        data = iris.data
        label = iris.target
        return data, label
    else:
        digits = load_digits()
        data = digits.data
        label = digits.target
        return data, label

def vega_scatter(df):
    # df = pd.DataFrame(data = [[1,2,1,'train'],[1,3,2,'train'], [3,4,3,'test']],
    #                 columns=['x', 'y', 'c', 'T'])
    circle_type={
        "mark":"circle",
        'height':400,
        'width':700,
        "encoding":{
            "x":{"field":'x', 'type':'quantitative'},
            'y':{'field':'y', 'type':'quantitative'},
            'color':{'field':"c", "type":"nominal"}
        },
        "selection": {
            "grid": {"type": "interval", "bind": "scales"}
        }         
    }
    st.vega_lite_chart(df, circle_type)

def process(alo_type, data, label):
    st.write(alo_type, data.shape)
    alo = None
    if alo_type == 'pca':
        alo = PCA(n_components=2)
    elif alo_type == 'tsne':
        alo = TSNE(n_components=2)
    elif alo_type == 'lle':
        alo = LocallyLinearEmbedding(n_components=2)
    
    data2D = alo.fit_transform(data)
    df = np.hstack([data2D, label.reshape(-1, 1)])
    df = pd.DataFrame(data = df, columns = ['x', 'y', 'c'])
    return df


if __name__ == "__main__":
    st.markdown('# streamDemo')
    st.markdown('---')
    sidebar = st.sidebar
    alo_type = sidebar.selectbox("可视化算法",('pca', 'tsne', 'lle'))
    data_type = sidebar.selectbox("数据集", ('iris', 'digits'))
    data, label = load_data(data_type)
    st.dataframe(data)
    st.markdown('---')
    if sidebar.button('run'):
        df = process(alo_type, data, label)
        vega_scatter(df)
