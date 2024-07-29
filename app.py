import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import plotly.express as px

def main():
    st.set_page_config(layout="wide",
                       initial_sidebar_state='collapsed')
    st.title('Sklearn learning app')
    numero_clusters = st.slider('Numero de clusters', 2, 10, 5)
    df = load_data()
    df, kmeans, inertias = cluster_dataframe(df, n_clusters=numero_clusters)
    st.plotly_chart(px.line(inertias))
    promedios = calcular_promedios_por_cluster(df, kmeans)
    plot_color(promedios)


def plot_color(promedios):
    inversed_colorscale = px.colors.diverging.RdBu[::-1]
    fig = px.imshow(promedios, color_continuous_scale=inversed_colorscale, width=1000)
    st.plotly_chart(fig)


def calcular_promedios_por_cluster(df: pd.DataFrame, kmeans: Pipeline):
    cols_sin_cluster = [col for col in df.columns if 'cluster' not in col]
    scaled_df = df.copy()
    for col in cols_sin_cluster:
        scaled_df[col] = (df[col] - df[col].mean())/df[col].std()
    
    return scaled_df.groupby('cluster').mean()


@st.cache_data()
def cluster_dataframe(df: pd.DataFrame, n_clusters: int = 5):
    
    pipelines_kmeans = {k: Pipeline([
        ('scaler', StandardScaler()),  # Scale data to have mean=0 and variance=1
        ('kmeans', KMeans(n_clusters=k))
    ]) for k in range(2, 10)}

    inertias = [kmeans['kmeans'].inertia_ for k, kmeans in pipelines_kmeans.items()]

    model = pipelines_kmeans[n_clusters].fit(df)
    df['cluster'] = model.predict(df)
    
    return df, pipelines_kmeans, inertias



@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet('data/transport_model.parquet')
    variables_bloque = ["block_CU", "block_AU", "block_AG", "block_FE", 
                        "block_S", "block_HG_OC", "block_CURE", "block_AURE", 
                        "block_PYCPY", "block_BIND", "block_RQD", "block_PY"]
    return df[variables_bloque]#.drop_duplicates()

    

if __name__ == '__main__':
    main()