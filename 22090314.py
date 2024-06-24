#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import errors as err
import cluster_tools as ct
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
pd.options.mode.chained_assignment = None


# In[2]:


def load_csv_data(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.

    Parameters:
    ------------    
    file_path (str): The filename of the CSV file to be read.

    Returns:
    ---------    
    df (pandas.DataFrame): The DataFrame containing the data 
    read from the CSV file.
    """
    address = file_path
    print(address)
    df = pd.read_csv(address, skiprows=4)
    df = df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 67'])
    return df


# In[3]:


def plot_kmeans_clusters(data, feature_x, feature_y, xlabel, ylabel, title, num_clusters, data_fit, data_min, data_max):
    """
    Plots a scatter graph of clusters using the KMeans algorithm.

    Parameters:
    ------------
    data (DataFrame): The DataFrame containing the data to cluster.
    feature_x (str): The name of the DataFrame column to plot on the x-axis.
    feature_y (str): The name of the DataFrame column to plot on the y-axis.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    title (str): The title of the plot.
    num_clusters (int): The number of clusters to form.
    data_fit (DataFrame): The DataFrame on which the KMeans will fit.
    data_min (float): Minimum scaling value used for backscaling the data.
    data_max (float): Maximum scaling value used for backscaling the data.

    Returns:
    ---------
    None
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
    kmeans.fit(data_fit)
    
    # Extract labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(data[feature_x], data[feature_y], c=labels, cmap="tab20")
    
    # Rescale and show cluster centers
    scaled_centers = ct.backscale(centers, data_min, data_max)
    x_centers = scaled_centers[:,0]
    y_centers = scaled_centers[:,1]
    plt.scatter(x_centers, y_centers, c="k", marker="d", s=80)
    
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('Clustering_plot.png', dpi=300)
    plt.show()


# In[4]:


def poly(x, a, b, c):
    """
    Calulates polynominal
    """
    x = x - 1990
    f = a + b*x + c*x**2

    return f


# In[5]:


def extract_country_data_over_period(dataframe, country, start_year, end_year):
    """
    Extracts and transforms data for a specified country and period from a given DataFrame.

    Parameters:
    ------------
    dataframe (DataFrame): The input DataFrame with countries as columns and years as rows.
    country (str): The name of the country for which data is to be extracted.
    start_year (int): The starting year for the data extraction.
    end_year (int): The ending year for the data extraction, inclusive.

    Returns:
    ---------
    DataFrame: A transformed DataFrame with year indices and data for the specified country and period.
    """
    # Transpose the DataFrame to switch rows and columns
    transposed_df = dataframe.T
    
    # Set the first row as column headers
    transposed_df.columns = transposed_df.iloc[0]
    
    # Drop the row used as the new header
    transposed_df = transposed_df.drop(['Country Name'])
    
    # Select only the column corresponding to the specified country
    country_data = transposed_df[[country]]
    
    # Convert index to integer for proper year filtering
    country_data.index = country_data.index.astype(int)
    
    # Filter data by the specified year range
    filtered_data = country_data[(country_data.index > start_year) & (country_data.index <= end_year)]
    
    # Ensure the data is of type float
    filtered_data[country] = filtered_data[country].astype(float)
    
    return filtered_data


# In[6]:


def plot_silhouette_score(data, max_clusters=10):
    """
    Evaluate and plot silhouette scores for different numbers of clusters.

    Parameters:
    - data: The input data for clustering.
    - max_clusters: The maximum number of clusters to evaluate.

    Returns:
    """

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color = 'yellow')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


# In[7]:


CO2_emissions_metric_tons_per_capita = load_csv_data('CO2_emissions_metric_tons_per_capita.csv')
Forest_area_of_land_area = load_csv_data('Forest_area_%_of_land_area.csv')
country = 'India'
df_co2 = extract_country_data_over_period(CO2_emissions_metric_tons_per_capita,country, 1990, 2020)
df_FA = extract_country_data_over_period(Forest_area_of_land_area,country, 1990, 2020)


# In[8]:


df = pd.merge(df_co2, df_FA, left_index=True, right_index=True)
df = df.rename(columns={country+"_x": 'Co2 Emissions Per Capita', country+"_y": 'Forest_Area'})
df_fit, df_min, df_max = ct.scaler(df)
plot_silhouette_score(df_fit,12)


# In[9]:


plot_kmeans_clusters(df,'Co2 Emissions Per Capita','Forest_Area','Co2 Emissions Per Capita','Forest Area','Co2 Emissions Per Capita vs Forest Area in India',2,df_fit,df_min,df_max)


# In[13]:


popt, pcorr = opt.curve_fit(poly, df_FA.index, df_FA[country])
# much better
df_FA["pop_poly"] = poly(df_FA.index, *popt)
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data",color = 'lightsteelblue')
plt.plot(df_FA.index, df_FA["pop_poly"], label="fit", color = 'limegreen')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in India 1990-2020')
plt.savefig(country+'_.png', dpi=300)

years = np.linspace(1990, 2030)
pop_ploy = poly(years, *popt)
sigma = err.error_prop(years, poly, popt, pcorr)
low = pop_ploy - sigma
up = pop_ploy + sigma
plt.figure()
plt.plot(df_FA.index, df_FA[country], label="data", color = 'lightsteelblue')
plt.plot(years, pop_ploy, label="Forecast")
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color='limegreen')
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Forest Area')
plt.title('Forest Area in India Forecast')
plt.savefig(country+'__forecast.png', dpi=300)
plt.show()


# In[15]:


popt, pcorr = opt.curve_fit(poly, df_co2.index, df_co2[country])
# much better
df_co2["pop_poly"] = poly(df_co2.index, *popt)
plt.figure()
plt.plot(df_co2.index, df_co2[country], label="data", color = 'lightsteelblue')
plt.plot(df_co2.index, df_co2["pop_poly"], label="fit", color = 'limegreen')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Co2 Emissions Per Capita')
plt.title('Co2 Emissions Per Capita in India 1990-2020')
plt.savefig(country+'_.png', dpi=300)
years = np.linspace(1990, 2030)
pop_poly = poly(years, *popt)
sigma = err.error_prop(years, poly, popt, pcorr)
low = pop_poly - sigma
up = pop_poly + sigma
plt.figure()
plt.plot(df_co2.index, df_co2[country], label="data", color = 'lightsteelblue')
plt.plot(years, pop_poly, label="Forecast")
# plot error ranges with transparency
plt.fill_between(years, low, up, alpha=0.5, color='limegreen')
plt.legend(loc="upper left")
plt.xlabel('Years')
plt.ylabel('Co2 Emissions Per Capita')
plt.title('Co2 Emissions Per Capita in India Forecast')
plt.savefig(country+'__forecast.png', dpi=300)
plt.show()


# In[ ]:




