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
    df = load_data()
    df, kmeans = cluster_dataframe(df)
    st.write(kmeans['scaler'])
    promedios = calcular_promedios_por_cluster(df, kmeans)
    st.write(promedios)
    plot_color(promedios)

    


def plot_color(promedios):
    fig = px.imshow(promedios)
    st.plotly_chart(fig)


def calcular_promedios_por_cluster(df: pd.DataFrame, kmeans: Pipeline):
    scaler = kmeans['scaler']
    cols_sin_cluster = [col for col in df.columns if 'cluster' not in col]
    df_scaled = pd.DataFrame(scaler.transform(df[cols_sin_cluster]), columns=)
    st.write(df_scaled)
    #pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df.groupby('cluster').mean()


def cluster_dataframe(df: pd.DataFrame, n_clusters: int = 5):
    
    kmeans = Pipeline([
        ('scaler', StandardScaler()),  # Scale data to have mean=0 and variance=1
        ('kmeans', KMeans(n_clusters=n_clusters))
    ])
    model = kmeans.fit(df)
    df['cluster'] = model.predict(df)
    
    return df, kmeans



@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet('data/transport_model.parquet')
    variables_bloque = ["block_CU", "block_AU", "block_AG", "block_FE", 
                        "block_S", "block_HG_OC", "block_CURE", "block_AURE", 
                        "block_PYCPY", "block_BIND", "block_RQD", "block_PY"]
    return df[variables_bloque]#.drop_duplicates()

    

if __name__ == '__main__':
    main()