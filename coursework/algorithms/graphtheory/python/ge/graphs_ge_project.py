
# coding: utf-8

# In[1]:

import pandas as pd
import scipy as sc
import numpy as np
from pprint import pprint
pd.set_option('display.max_columns', 50)


# In[2]:

def get_data(filename):
    # Acquire data
    df = pd.read_csv(filename)
    n_samples, n_features = df.shape
    print "Num Samples: {0}, Num Features: {1}".format(n_samples, n_features)
    return df

def prepare_data(df_in):
    # factorize categorical data
    df = df_in
    df.X9 = df.X9.astype('category')
    df_cat = pd.get_dummies(df["X9"], prefix="fe")
    df = df.join(df_cat)
    df = df.drop(['X9'], axis=1)
    # for now drop the categorical features from get_dummies
    # df = df.drop(list(df_cat.columns), axis=1)

    # drop additional columns for featurization: [X15,X16,X17,X18,X19]
    # df = df.drop(["X15","X16","X17","X18","X19"], axis=1)

    # imputate and convert to numeric - replace 'null' string values with 'nan'
    for index, row in df.iterrows():
        for c in row.index:
            if row[c] == "null": df.ix[index, c] = np.nan
    # find the median of each feature and replace any null values with these             
    feature_imputate = {}
    for c in list(df.columns): feature_imputate[c] = df[c].median()
    # imputate median values and set as numeric 
    for c in list(df.columns):
        df[c] = df[c].fillna(feature_imputate[c])
        df[c] = df[c].astype(np.float)
    # Scale the matrix via min-max
    df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))    
    return df
    
df_raw = get_data('./ge_data.csv')
df = prepare_data(df_raw)
#df_raw.head()
df.head()


# In[3]:

import scipy.spatial.distance as dist
def calculate_similarity_matrix(df_in):
    # calculate pairwise squared euclidean distances
    dist_mat = dist.pdist(df_in.values)
    dist_sq_matrix = dist.squareform(dist_mat)
    df_dist = pd.DataFrame(dist_sq_matrix)
    # replace diagonal with 1's as they should be identical
    # values of 1 for indices close and values of indices far apart
    np.fill_diagonal(df_dist.values, 1)
    # invert the scores and add a small number (epsilon), to avoid division by zero
    col = list(df_dist.columns)    
    adj_sq_matrix = df_dist.ix[:,col].apply(lambda w: 1/(w + 1e-6))    
    print "Dimensions - Dataframe:{0}, Distance Matrix:{1}, Adj Matrix:{2}".format(df_in.shape, dist_sq_matrix.shape, adj_sq_matrix.shape)
    return adj_sq_matrix

def check_samples_symmetric(df_in, i,j):
    # verify the matrix is symmetric via particular cell [i][j] = [j][i]
    return (df_in[i][j] == df_in[j][i])

similarity_matrix = calculate_similarity_matrix(df)
print "Symmetric Matrix State: {0}".format(check_samples_symmetric(similarity_matrix, 246,55))
similarity_matrix.head(5)


# In[4]:

# Compute the Laplacian Eigenvectors/Eigenvalues
def calculate_laplacian_eigen(df_in):    
    D = []
    # sum the rows and form into a diagonal
    for idx, v in df_in.iterrows(): D.append(sum(v.values))
    D = np.diag(D)
    D = np.array(D)
    # compute the Laplacian Matrix
    I  = np.identity(len(df_in))
    # compute a normalized eigenvectors of the Laplacian matrix
    # L = D - df_in.values
    L = I - np.dot(D,np.dot(df_in.values,D))

    # each column returned in (V) is an eigenvector, use the fielder vector (second lambda) as the optimal cut
    # as we have verified the input matrix is already symmetric, we use the symmetric version
    return sc.linalg.eigh(L)

# Compute Fieldler Vector (lambda,2) = first non-trivial and rescale it such that the values are on a [0..1] scale
def compute_fieldervector(eigen_v, max_k):
    # Fielder Vector: based on algebaic connectivity
    # algebraic connectivity -> 2nd smallest eigenvalue: connected graph only if > 0
    # number of times 0 appears as an eigenvalue in the Laplacian is indication of number of connected components in the graph

    # Alternatively use a lower dimension of the eigenvectors of k clusters (nxk) dimensions
    # Transpose to columns
    U = eigen_v[:max_k].T
    fieldler_vector = eigen_v[:,1].T
    return pd.DataFrame(fieldler_vector, columns=['eigenvector'])

def rescale(df_in):
    # normalize all fiedler vectors between 0 and 1 scale: (x - xmin) / (xmax - xmin)
    xmin = df_in.min()
    xmax = df_in.max()
    rescaled_vector = np.array([ (x -xmin)/(xmax-xmin) for x in df_in.values])
    return rescaled_vector

# selection of k: determine largest delta gap of eigenvector, maximizes k = |lambda(k) - lambda(k-1)| 
def find_ideal_cluster_num(eigen_val_v):
    k_delta = [0]
    for i in range(1, len(eigenvalues)-1): k_delta.append(abs(eigenvalues[i] - eigenvalues[i-1]))
    ind = np.argsort(k_delta)[::-1]
    largest_sep = [(i,k_delta[i]) for i in ind]
    return ind, largest_sep
   
# the implementation chosen is based on k-partitioning on the fielder vector
def create_clusters(fv_in, k_min, k_max):
    Ci = []
    xmin = fv_in.min()[0]
    xmax = fv_in.max()[0]
    thresholds = [-0.25, -0.1e-5, -0.1e-6, -0.1e-7, -0.05e-7, -0.1e-8, -0.05e-8, -0.015e-8, -0.025e-8,
                  -0.1e-9, -0.25e-8, -0.1e-10, -0.1e-11, -0.1e-12, 0.0001, 0.001, 1]
    thresholds = sorted(thresholds)
    
    for k in range(k_min,k_max+1):
        threshold_client = []
        prev_thresh = xmin - 1e-6
        sumx = 0
        # this is how the thresholds will be set for the cluster selection
        # from the threshold list, determine which clusters to select based on partitioning these clusters
        # Manually setting a threshold list from above, rather than calculated due to lack of differentiation
        for idx, t in enumerate(thresholds):
            if (len(threshold_client) == (k-1)): 
                ind = list(fv_in[(fv_in.eigenvector > thresholds[idx-1]) & (fv_in.eigenvector > prev_thresh)].index)            
            else:
                ind = list(fv_in[(fv_in.eigenvector < t) & (fv_in.eigenvector > prev_thresh)].index)
            
            prev_thresh = t
            if (len(ind) > 0): threshold_client.append(ind)
            if(len(threshold_client) == k):break
        Ci.append(threshold_client) 
    return Ci
    
# retrieve eigenvectors/eigenvalues
eigenvalues, eigenvectors = calculate_laplacian_eigen(similarity_matrix)  
# rescale the eigenvectors
df_fieldler = compute_fieldervector(eigenvectors, 10)
# the largest separation of predecessor eigenvalues should demonstrate the optimal value of k
k_list, _ = find_ideal_cluster_num(eigenvalues)
print "Top 10 Optimal values of k, based on separation ",k_list[:10] 
# for each cluster size (k), create partitions based on the rescaled partitions
clusters_l = create_clusters(df_fieldler, 2, 10)


# In[5]:

# Log cluster results to files
import os
# log indices grouped together to a file for the particular cluster size
# these are only the indices, not the data itself
def log_indices(clusters_in, filename):
    for k_idx, k_vector in enumerate(clusters_in):
        with open(filename + str(k_idx+2) + ".txt", 'w') as f: 
            for i in range(len(k_vector)): 
                header = "[Cluster__" + str(i+1) + "]" + "\r\n"
                output = header
                k_vector_sorted = sorted(k_vector[i])
                for w in k_vector_sorted: output += str(w) + ", "
                output = output.rstrip(", ")
                output += "\r\n\n"
                f.write(output)    

directory = os.path.dirname("./output/")
if not os.path.exists(directory):os.makedirs(directory)
# log the cluster results from k=2 to 10 to separate text files
log_indices(clusters_l, "./output/clustersize_")


# In[6]:

# format data to create for xlsx format for better analysis and viewing
def format_for_xlsx(Ci):
    k_df = []
    for k_idx, k_vector in enumerate(Ci):
        header_rowsize = 1
        k_offset = [header_rowsize]
        numrows = header_rowsize
        df_matched = pd.DataFrame()
        for i in range(len(k_vector)):
            # find the original dataframe index that matches
            k_vector_sorted = sorted(k_vector[i])
            df_matched = df_matched.append(df_raw.ix[k_vector_sorted, :])
            # keep track of the dataframe so we can can update the xlsx all together        
            numrows += len(k_vector_sorted)
            k_offset.append(numrows)
        df_matched.index.name = "Index"
        k_df.append([df_matched, k_offset])
    return k_df

# output to *.xlsx file
def write_xlsx(cluster_format, filename):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    colors = ['cyan', '#FFFFCC', '#CCFFFF', '#FFCC99', '#C0C0C0', '#FFFF99', '#33CCCC', '#CCCCFF', '#FF99CC', '#CCFFCC']
    for k_cluster, df_w in enumerate(cluster_format):
        sheet_label = 'k=%s' % str(k_cluster+2)
        df_w[0].to_excel(writer, startcol=1, sheet_name=sheet_label)
        worksheet = writer.sheets[sheet_label]
        # update what cluster this is (column 0)
        for idx in range(len(df_w[1])-1): 
            worksheet.write_string(df_w[1][idx], 0, "Cluster__" + str(idx+1))    
            # iterate over the row selection and change the color for the format
            format_style = writer.book.add_format()
            format_style.set_bg_color(colors[idx])   
            for row in range(df_w[1][idx],df_w[1][idx+1]):worksheet.set_row(row, None, format_style)
        #t_fmt = writer.book.add_format({'num_format': '$#,##0'})
        #worksheet.set_column('D:D', None, t_fmt)
    writer.save()
    
# format, match original indexes for data and write the data to *.xlsx
xls_frames = format_for_xlsx(clusters_l)
write_xlsx(xls_frames, './ge_spectralcluster_analysis.xlsx')


# In[7]:

# plot the values and of eigenvalues with larges separation for selection of k
# %matplotlib inline
# import matplotlib.pyplot as plt


# In[ ]:



