import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Main function to set up the Streamlit app
def main():
    st.set_page_config(layout="wide", initial_sidebar_state='collapsed')  # Configuring the Streamlit app layout
    st.title('Learning app without sklearn')  # Setting the title of the app
    df = load_data()  # Loading the dataset
    df, scaler, centroids = cluster_dataframe(df)  # Clustering the dataset
    st.write(scaler)  # Displaying the scaler parameters (mean and std)
    scaled_df, mean, std = standard_scale(df)
    promedio = calcular_promedios_por_cluster(scaled_df)  # Calculating cluster means
    st.write(promedio)  # Displaying the cluster means
    plot_color(promedio)

# Function to plot the cluster means using Plotly
def plot_color(promedios):
    fig = px.imshow(promedios)  # Creating a heatmap
    st.plotly_chart(fig)  # Displaying the heatmap in Streamlit

# Function to calculate the mean values for each cluster
def calcular_promedios_por_cluster(df: pd.DataFrame):
    return df.groupby('cluster').mean()  # Grouping the DataFrame by 'cluster' and calculating the mean

# Function to standardize the dataset
def standard_scale(df: pd.DataFrame):
    mean = df.mean()  # Calculating the mean of each column
    std = df.std()  # Calculating the standard deviation of each column
    scaled_df = (df - mean) / std  # Standardizing the DataFrame
    return scaled_df, mean, std  # Returning the standardized DataFrame, mean, and std

# Function to perform K-means clustering without sklearn
def kmeans_clustering(df: pd.DataFrame, n_clusters: int, n_init: int = 10, max_iter: int = 300):
    best_inertia = None
    best_centroids = None
    best_labels = None
    
    for _ in range(n_init):  # Running the clustering process multiple times to find the best initialization
        centroids = df.sample(n_clusters).to_numpy()  # Randomly selecting initial centroids
        for _ in range(max_iter):  # Iterating to update centroids
            distances = np.linalg.norm(df.to_numpy()[:, np.newaxis] - centroids, axis=2)  # Calculating distances from data points to centroids
            labels = np.argmin(distances, axis=1)  # Assigning labels based on closest centroid
            new_centroids = np.array([df.to_numpy()[labels == i].mean(axis=0) for i in range(n_clusters)])  # Updating centroids
            
            if np.all(centroids == new_centroids):  # Checking for convergence
                break
            centroids = new_centroids
        
        inertia = np.sum((df.to_numpy() - centroids[labels]) ** 2)  # Calculating inertia (sum of squared distances)
        
        if best_inertia is None or inertia < best_inertia:  # Checking if this is the best initialization so far
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels
    
    return best_labels, best_centroids  # Returning the best labels and centroids

# Function to cluster the dataset and return the scaled DataFrame and cluster assignments
def cluster_dataframe(df: pd.DataFrame, n_clusters: int = 5):
    scaled_df, mean, std = standard_scale(df)  # Standardizing the dataset
    labels, centroids = kmeans_clustering(scaled_df, n_clusters)  # Performing K-means clustering
    df['cluster'] = labels  # Adding the cluster labels to the original DataFrame
    return df, (mean, std), centroids  # Returning the DataFrame with clusters, scaler parameters, and centroids

# Function to load the dataset
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet('data/transport_model.parquet')  # Reading the dataset from a parquet file
    variables_bloque = ["block_CU", "block_AU", "block_AG", "block_FE", 
                        "block_S", "block_HG_OC", "block_CURE", "block_AURE", 
                        "block_PYCPY", "block_BIND", "block_RQD", "block_PY"]  # Selecting relevant columns
    return df[variables_bloque]  # Returning the selected columns

# Entry point of the script
if __name__ == '__main__':
    main()  # Running the main function