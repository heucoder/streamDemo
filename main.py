from LE import le
from LLE import lle
from ISOMAP import my_Isomap as isomap
import streamlit as st
from sklearn.datasets import make_s_curve
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache
def load_data(n_samples=500, noise=0.1, random_state=42):
    X, Y = make_s_curve(n_samples, noise, random_state)
    return X, Y

def plot_3d(X, Y):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.hot)
    # ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    st.pyplot()

def plot_2d(df, Y):
    c = alt.Chart(df).mark_point().encode(
        x='x',
        y='y',
        color='class'
    ).interactive()
    st.altair_chart(c)

if __name__ == "__main__":
    st.title("Streamlit Demo")
    sidebar = st.sidebar
    res = sidebar.selectbox("选择一种降维算法", ("le", "lle", "isomap"))
    n_neighbors = sidebar.slider('neighbors', 5, 20, 5, 1)
    t = sidebar.slider('t', 1, 10, 1, 1)
    X, Y = load_data()

    plot_3d(X, Y)

    start_button = sidebar.button("BEGIN")
    if start_button:
        if res == 'le':
            st.write(res, n_neighbors, t)
            data = le(X, 2, n_neighbors, t)
            data = np.hstack([data, Y.reshape(-1, 1)])
            df = pd.DataFrame(data, columns=['x', 'y', 'class'])
            plot_2d(df, Y)
        elif res == 'lle':
            st.write(res, n_neighbors, t)
            data = lle(X, 2, n_neighbors)
            data = np.hstack([data, Y.reshape(-1, 1)])
            df = pd.DataFrame(data, columns=['x', 'y', 'class'])
            plot_2d(df, Y)
        elif res == 'isomap':
            st.write(res, n_neighbors, t)
            data = isomap(X, 2, n_neighbors)
            data = np.hstack([data, Y.reshape(-1, 1)])
            df = pd.DataFrame(data, columns=['x', 'y', 'class'])
            plot_2d(df, Y)
