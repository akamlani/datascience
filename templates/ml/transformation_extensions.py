from sklearn.decomposition import PCA, RandomizedPCA

def pca_analysis(X,N):
    '''pca analysis: X=input an array style data set, N=Number of Components to analyze''' 
    n = N
    comp_id=[]
    for i in range(1,n+1): comp_id.append(i)      #id number of each component
    pca = PCA(n_components = n)
    X_r = pca.fit(X).transform(X)
    
    ratios = pca.explained_variance_ratio_
    sum_ratios = pca.explained_variance_ratio_.sum()
    print "Ratios of Variance Explained: ", ratios
    print "Total Variance Explained: ", sum_ratios
    
    #Scree Plot
    fig = plt.figure(figsize=(8,5))
    plt.plot(comp_id, ratios, 'ro-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')

