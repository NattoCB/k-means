
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    # standard k-means
    def __init__(self, data=None, k=2, min_gain=0.01, max_iter=1):
        if data is not None:
            self.fit(data, k, min_gain, max_iter, max_itr)

    def fit(self, data, k=2, min_gain=0.01, max_iter=100, max_itr=1):
        # Pre-process
        self.data = np.matrix(data)
        self.k = k
        self.min_gain = min_gain

        # Perform multiple random init for global optimum
        min_sse = np.inf
        for i in range(max_itr):

            # Randomly initialize k centroids
            indices = np.random.choice(len(data), k, replace=False)
            cent = self.data[indices, :-1]
            itr = 0 # count iteartion times
            old_sse = np.inf # initialise
            while True:
                itr += 1
                # Cluster assignment
                C = [None] * k
                for sample in self.data:
                    j = np.argmin(np.linalg.norm(sample[:,:-1] - cent, 2, 1))
                    C[j] = sample if C[j] is None else np.vstack((C[j], sample))
                # Centroid update
                for j in range(k):
                    cent[j] = np.mean(C[j][:,:-1],0)
                # Loop termination condition
                if itr >= max_iter:
                    break
                new_sse = np.sum([sse(C[j]) for j in range(k)])
                gain = old_sse - new_sse

                if gain < self.min_gain:
                    if new_sse < min_sse:
                        min_sse, self.C, self.cent = new_sse, C, cent
                    break
                else:
                    old_sse = new_sse

        return self


class BisectingKMeans:
    # Bisecting k-means internally uses std k-means with k=2
    
    def __init__(self, data, max_k=10, min_gain=0.1):
        if data is not None:
            self.fit(data, max_k, min_gain)

    def fit(self, data, max_k=10, min_gain=0.1):
        # Learns from given data and options

        self.kmeans = KMeans()
        self.C = [data, ] # a list of cluster,and data 
        self.m = data.shape[0]    # total samples
        self.n = data.shape[1]    # total features
        self.k = len(self.C)      # total clusters
        self.u = np.reshape(      # centroids for each cluster
            [np.mean(self.C[i],0) for i in range(self.k)], (self.k, self.n))

        while True:
            # pick a cluster to bisect
            sse_list = [sse(data) for data in self.C]
            old_sse = np.sum(sse_list)
            data = self.C.pop(np.argmax(sse_list))
            # bisect it
            self.kmeans.fit(data, k=2)
            # add bisected clusters to our list
            self.C.append(self.kmeans.C[0])
            self.C.append(self.kmeans.C[1])
            self.k += 1
            self.u = np.reshape([np.mean(self.C[i],0) for i in range(self.k)], (self.k, self.n))
            # check SEE and k
            sse_list = [sse(data) for data in self.C]
            new_sse = np.sum(sse_list)
            gain = old_sse - new_sse
            # min_gain: Minimum gain to keep iterating
            if gain < min_gain or self.k >= max_k:
                break
        return self

'''
    -----------------------------  data loading ---------------------------- 
                                                                             '''

def get_data(dataPath,label):
    # data loading
    dataVolume = len(open(dataPath,'r').readlines()) # total lines
    data = open(dataPath,'r')
    dataSet = []
    for i in range(0, dataVolume):
        j = data.readline().replace('\n','').split(' ')
        floatData = [float(i) for i in j[1:]]
        dataSet.append(floatData)
    dataArray = np.asarray(dataSet)
    labels = np.linspace(label,label,dataVolume)
    finalData = np.column_stack((dataArray,labels))
    return finalData

def dataGet_Integrate(norm=False):
    # get data  (300 dims of features, last 1 dim for lables) and integrate together
    dataSet1 = get_data('data/animals',  1)
    dataSet2 = get_data('data/fruits',   2)
    dataSet3 = get_data('data/veggies',  3)
    dataSet4 = get_data('data/countries',4)
    partA = np.row_stack((dataSet1,dataSet2))
    partB = np.row_stack((dataSet3,dataSet4))
    allData = np.row_stack((partA,partB))
    if norm == True: # do l-2 normalisation
        d = allData[:,:-1]
        for i in range(0,329): 
            norm = np.sqrt(np.sum(d[i,:] * d[i,:]))
            d[i,:] = d[i,:] / norm
        normData = np.column_stack((d,allData[:,-1])) 
        return normData # normalised
    else:
      return allData # unnormalised

'''
    ---------------------------  math functions ---------------------------- 
                                                                             '''
def sse(data):
    # SSE calculation
    cent = np.mean(data, 0)
    return np.sum(np.linalg.norm(data - cent, 2, 1))

def get_target_pred_comb(result):
    # return a result arrary like [target, pred]
    result_pred = []
    for i,j in enumerate(result):
        comb = np.column_stack((np.linspace(i,i,j.shape[0]),j[:,-1]))
        for z in comb:
            result_pred.append([z[0,0],z[0,1]])
    r_p_arr = np.asarray(result_pred)
    return r_p_arr

def get_coocMatrix(sorted_comb,k):
    # for calculate TP TN FP FN
    cooc_matrix = [] # generate a zero matrix (2d-list) for future update
    for i in range(4):
        cooc_matrix.append([])
        for j in range(k):
            cooc_matrix[i].append(0)

    d1,d2 = sorted_comb.shape # get all data volume
    current_k = 0 # initialise k_stat
    for i in range(d1):  # iter all data to update cooc-matrix
        current_c = int(sorted_comb[i,1])
        cooc_matrix[current_c-1][int(sorted_comb[i,0])]+=1
    cm = np.asarray(cooc_matrix) # arralisation    
    return cm # Co-oc Metrix

def get_tp_tn_fp_fn(cooccurrence_matrix):
    # calculate TP, TN, FP, FN by construct a co-oc metrix
    tp_plus_fp = my_vComb(cooccurrence_matrix.sum(0,dtype=int),2).sum() # sum dim-0
    # calculate Comb(dim-0  2) then add together
    tp_plus_fn = my_vComb(cooccurrence_matrix.sum(1,dtype=int),2).sum() # sum dim-1
    # calculate Comb(dim-1  2) then add together
    
    tp = my_vComb(cooccurrence_matrix.astype(int),2).sum() 
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = my_vComb(cooccurrence_matrix.sum(),2) - tp - fp - fn # C(n  2)
    
    return [tp,tn,fp,fn]

def get_p_r_f(cooccurrence_matrix):
    # get precision, recall, and F1 measure
    tp,tn,fp,fn = get_tp_tn_fp_fn(cooccurrence_matrix)
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = (2*p*r)/(p+r)
    return [p,r,f]

def get_purity(cooccurrence_matrix):
    # calculate purity
    a=0
    for i in cooccurrence_matrix.T:
        a += np.amax(i)
    return a/329

# calculate factorial
def my_factorial(a):
    b = int(a)
    for i in range(0,int(a)-1):
        b = b * (int(a)-1)
        a -= 1
    return b

# calculate combination C(n r)
def my_comb(n,r): 
    if isinstance(n,int) == True: # input is an integer
        try:
            return my_factorial(n)/(my_factorial(r)*my_factorial(n-r))
        except ZeroDivisionError as e:
            return 0
    else: # processing inputs as an array
        nList = []
        for i in range(n):
            try:
                with np.errstate(divide='ignore'):
                    n_ = my_factorial(i)/(my_factorial(r)*my_factorial(i-r))
            except ZeroDivisionError as e:
                n_ = 0
            nList.append(n_)
        nArr = np.asarray(nList)
        return nArr
my_vComb = np.vectorize(my_comb) # as vectors

def visualise(P,R,F,K,Purity_List):
    # plot
    plt.figure(figsize=(8,4)) 
    plt.plot(K,P,'b--',linewidth=1,label='precision')  
    plt.plot(K,R,'r--',linewidth=1,label='recall') 
    plt.plot(K,F,'y--',linewidth=1,label='f1-score') 
    plt.plot(K,Purity_List,'g--',linewidth=1,label='purity')    
    plt.xlabel("k-clusters")
    plt.title('Accuracy plot of Bisecting K-Means')
    plt.legend(loc='upper right')
    plt.show()

'''
    ------------------------------  main part ------------------------------ 
                                                                             '''

def do_k_2_to_10_test():
    # do k=2 to 10 test
    P_List,F_List,R_List,K_List,Purity_List = [],[],[],[],[]
    
    for k in range(2,11):
        c = BisectingKMeans(data1, k) # as a new BKMs class
        r_p_arr = get_target_pred_comb(c.C) # get the result of [target, pred] array and co-oc metrix
        cm = get_coocMatrix(r_p_arr,k) # get a co-oc metrix of k-clusters and acturall 4-classes
        pur = get_purity(cm) # get purity
        p,r,f = get_p_r_f(cm) # get precision, recall, f1-score
        P_List.append(p)
        R_List.append(r)
        F_List.append(f)
        K_List.append(k)
        Purity_List.append(pur)
        print('Bisecting K-Means Result:\nPrecision: {}, Recall: {}, F1-score: {}, Purity: {}, when K={}'.
            format(round(p,5),round(r,5),round(f,5),round(pur,5),int(k)))
        print('The cooc_matrix when k={} is: \n{}\n'.format(k,cm))
    
    return K_List,P_List,R_List,F_List,Purity_List

if __name__ == '__main__':

    data1 = dataGet_Integrate(norm=False)
    K_List,P_List,R_List,F_List,Purity_List = do_k_2_to_10_test()
    visualise(P_List,R_List,F_List,K_List,Purity_List)








