# -*- coding: utf-8 -*-

#Importing all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import importlib

#These modules are from taken from class
import cluster_tools as ct
import errors as err


def read_data(file_path):
    '''
    This function reads data from given file path and cleans data to select 
    columns from 1960 to 2021.

    Parameters
    ----------
    file_path : STR
        filepath.

    Returns
    -------
    data : dataframe
        dataframe created using csv from filepath.

    '''
    data = pd.read_csv(file_path, skiprows=4)
    data = data.set_index('Country Name', drop=True)
    data = data.loc[:, '1960':'2021']

    return data


def transpose(data):
    '''
    This function creates transpose of given data

    Parameters
    ----------
    data : dataframe
        dataframe for which transpose to be found.

    Returns
    -------
    data_tr : dataframe
        transpose of given dataframe.

    '''
    data_tr = data.transpose()

    return data_tr


def corr_scattermatrix(data):
    '''
    This function prints correlation coefficient matrix of columns of given
    dataframe and scatter plots values.

    Parameters
    ----------
    data : dataframe
        dataframe for which analysis to be done.

    '''
    corr = data.corr()
    print('Correlation Coefficient matrix of given data')
    print(corr)

    plt.figure(figsize=(10, 10))
    plt.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation between Columns of given data')
    plt.colorbar()
    plt.savefig('corr.png', bbox_inches='tight', dpi=300)
    plt.show()

    pd.plotting.scatter_matrix(data, figsize=(12, 12), s=5, alpha=0.8)
    plt.savefig('scattter.png', bbox_inches='tight', dpi=300)
    plt.show()

    return


def n_clusters(data, data_norm, a, b):
    '''
    This function will find the number of clusters that are good fit for the 
    data using silhouette score.

    Parameters
    ----------
    data : dataframe
        dataframe with actual values.
    data_norm : dataframe
        dataframe with normalized values.
    a : STR
        Column 1.
    b : STR
        Column 2.

    Returns
    -------
    INT
        n cluster.

    '''
    n_clusters = []
    cluster_score = []
    print()
    print("n  score")
    # loop over number of clusters

    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(data_norm)
        labels = kmeans.labels_

        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(data, labels))

        n_clusters.append(ncluster)
        cluster_score.append(skmet.silhouette_score(data, labels))

    n_clusters = np.array(n_clusters)
    cluster_score = np.array(cluster_score)

    best_ncluster = n_clusters[cluster_score == np.max(cluster_score)]
    print()
    print('n cluster', best_ncluster[0], '\n')

    return best_ncluster[0]


def clusters_and_centers(data, ncluster, a, b):
    '''
    This function will create clusters for given data

    Parameters
    ----------
    data : dataframe
        dataframe for which clusters to be created.
    ncluster : INT
        number of clusters.
    a : STR
        Column 1.
    b : STR
        Column 2.

    Returns
    -------
    None.

    '''
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(data)
    labels = kmeans.labels_
    data['labels'] = labels
    cen = kmeans.cluster_centers_
    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(data[a], data[b], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"CO2 Emissions per capita({a})")
    plt.ylabel(f"CO2 Emissions per capita({b})")
    plt.title(f'CO2 Emissions per capita in {a} and {b}')
    plt.legend()
    plt.savefig('clusters.png', bbox_inches='tight', dpi=300)
    plt.show()

    print('Cluster Centers')
    print(cen, '\n')

    return


def logistic(t, n0, g, t0):
    """
    Using this function from class
    Calculates the logistic function with scale factor n0 
    and growth rate g
    """

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def forecast_co2(data, country):
    '''
    This function will fit data of a given country with logistic fitting used 
    in the class and forecast the CO2 Emissions per capita for year 2030.

    Parameters
    ----------
    data : dataframe
        transposed data of Worldbank data.
    country : STR
        Country for which forecast to be made.

    Returns
    -------
    None.

    '''
    data = data.loc[:, country]
    data = data.dropna(axis=0)

    co2_forecast = pd.DataFrame()
    co2_forecast['Year'] = pd.DataFrame(data.index)
    co2_forecast['CO2'] = pd.DataFrame(data.values)

    co2_forecast["Year"] = pd.to_numeric(co2_forecast["Year"])
    importlib.reload(opt)
    param, covar = opt.curve_fit(logistic,
                                 co2_forecast["Year"], co2_forecast["CO2"],
                                 p0=(1.2e12, 0.03, 1990.0))
    sigma = np.sqrt(np.diag(covar))

    year = np.arange(1990, 2031)
    forecast = logistic(year, *param)
    low, up = err.err_ranges(year, logistic, param, sigma)
    plt.figure()
    plt.plot(co2_forecast["Year"], co2_forecast["CO2"], label="CO2")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("CO2 Emissions per capita (USD)")
    plt.legend(loc='best')
    plt.title(f'CO2 Emissions per capita forecast for {country}')
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    # assuming symmetrie estimate sigma
    gdp2030 = logistic(2030, *param)/1e9
    low, up = err.err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print()
    print(f"CO2 Emissions per capita for {country} in 2030 is ",
          gdp2030*1e9, "+/-", sig)


#Reading data
co2 = read_data("co2_per_capita.csv")
print(co2.describe())

#Creating transpose for my data
co2_tr = transpose(co2)
print(co2_tr.head())

#Selecting few years for our analysis
co2 = co2[["1990", '1995', "2000", '2005', "2010", '2015', '2019']]
print(co2.describe())

#Correlation matrix of our selected data
corr_scattermatrix(co2)

a = "1990"
b = "2019"
co2_ab = co2[[a, b]]
co2_ab = co2_ab.dropna()
print(co2_ab.head())

# Normalising the data
co2_norm, co2_min, co2_max = ct.scaler(co2_ab)

#Selecting best number of clusters for our selected data
ncluster = n_clusters(co2_ab, co2_norm, a, b)

#Plotting Clusters and Centers for normalized data
clusters_and_centers(co2_norm, ncluster, a, b)

#Plotting clusters and centers for actual data
clusters_and_centers(co2_ab, ncluster, a, b)

#Plotting countries in the last cluster
print(co2_ab[co2_ab['labels'] == ncluster-1].index)

#Forecasting CO2 per capita in 2030 for Qatar
forecast_co2(co2_tr, 'Qatar')

#Forecasting CO2 per capita in 2030 for United States
forecast_co2(co2_tr, 'United States')

#Forecasting CO2 per capita in 2030 for UNited Kingdom
forecast_co2(co2_tr, 'United Kingdom')

#Forecasting CO2 per capita in 2030 for China
forecast_co2(co2_tr, 'China')
