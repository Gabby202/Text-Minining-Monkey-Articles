import nltk, re, math, pandas as pd, numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates

## contains useful methods to generate the word vectors/tf-idf matrix
def build_bow (tokenized_docs):    
    freq_dists = []
    for tokenized_doc in tokenized_docs:
        d = {}
        for token in tokenized_doc:
            if token in d.keys():
                d[token] += 1
            else:
                d[token] = 1
        freq_dists.append(pd.Series(d))
    return pd.DataFrame(freq_dists)

## iterative functions to build the tfidf matrix
def compute_totals(collection_of_vectors):
    total_counts = []
    for vector in collection_of_vectors:
        total = 0
        for  entry in vector:
            total += entry
        total_counts.append(total)
    return total_counts
def build_tf_bow (bow, total_counts):
    new_bow = pd.DataFrame(index=bow.index, columns=bow.columns)
    for i in range( len(total_counts)):
        total = total_counts[i]
        row = bow.iloc[i]
        for col in row.index:            
            tf_entry = row[col] / total
            new_bow.iloc[i][col] = tf_entry
    return new_bow
def compute_idfs(bow):
    N = len(bow.index)
    nis = []
    for ti in bow.columns:
        ni = 0
        for row in range(N):
            if bow[ti].iloc[row] > 0:
                ni += 1     
        nis.append(ni)
    return [math.log2(N/ni) for ni in nis]
def build_tfidf_bow (bow, idfs):
    docs_names = list(bow.index); terms = list(bow.columns)
    new_bow = pd.DataFrame(index= docs_names, columns=terms)
    for i in range( len(idfs)):
        idf = idfs[i]
        col = terms[i]        
        for name in docs_names:
            new_bow.loc[name][col] = bow.loc[name][col]  * idf
    return new_bow

## using vectorized operations on pandas frames... much more efficient than
## the iterative solutions above
def vectorized_compute_totals(df):   
    return df.sum(axis=1)
def vectorized_build_tf_bow(df, totals):
    return df.div(totals, axis=0)
def vectorized_compute_idfs(df):
    nis = df[df>0].count(axis=0)
    N = len(df)
    idfs = []
    for item in nis:
        idfs.append(math.log2(N/item))
    return idfs
def vectorized_build_tfidf_bow(df):
    totals = vectorized_compute_totals(df)
    tf_bow = vectorized_build_tf_bow(df, totals)
    idfs = vectorized_compute_idfs(df)
    return tf_bow.multiply(idfs, axis=1)

## clustering algs
def clustering_k_means(data, k=2):
    model = cluster.KMeans(k)
    model.fit(data)
    clust_labels = model.labels_ + 1##model.predict(data)
    centers = model.cluster_centers_
    return (clust_labels, centers)
def clustering_agglomerative(data):
    model = cluster.AgglomerativeClustering(
        affinity='cosine',linkage='single')
    model.fit(data)
    cluster_labels = model.labels_ + 1
    return cluster_labels

## visualising clustering
def clustering_scatter_plot(df, term1, term2):
    fig = plt.figure()
    ax = fig.add_subplot(133)
    scatter = ax.scatter(df[term1],df[term2],
                         c=df['cluster'],s=50)
    ax.set_title('Clustering')
    ax.set_xlabel(term1)
    ax.set_ylabel(term2)
    plt.colorbar(scatter)
    plt.show()
def display_k_centers(df, cluster_centers, cluster_labels):
    centers_df = pd.DataFrame(cluster_centers,
                    index=['Means1', 'Means2'],
                    columns=list(df.columns))
    centers_df['cluster'] = [1, 2]
    ##print(centers_df)
    plt.figure(figsize=(7, 5))
    plt.title('Clusters 1 and 2 means along 5 terms')
    parallel_coordinates(centers_df, 'cluster',
                         color=['blue', 'red'], marker='o')
    plt.show()    
def display_dendrogram(df, method='single'):
    Z = linkage(df, method)
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Documents/Observations')
    plt.ylabel('Distance')
    dendrogram(Z, labels=df.index, leaf_rotation=90)
    plt.show()

## visualizing classification
def classification_visualizaion(x_scores, xval_labels, split_scores,split_labels,
                                false_vals, false_labels, true_vals,true_labels):
    plt.figure(1, figsize=(25, 7))
    plt.subplot(341).set_title('Cross Validation')
    plt.bar(xval_labels, x_scores, color='green')
    plt.subplot(342).set_title('Split Validation')
    plt.bar(split_labels, split_scores, color='orange')
    plt.subplot(343).set_title('True Values')
    plt.bar(true_labels, true_vals, color='blue')
    plt.subplot(344).set_title('False Values')
    plt.bar(false_labels, false_vals, color='red')
    plt.suptitle('Classification Results for NB and SVM')
    plt.show()
