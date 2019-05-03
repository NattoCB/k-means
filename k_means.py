import numpy as np
import matplotlib.pyplot as plt

c = 4 # fruits, veggies, countries, animals (4-class labels)
m = 329 # total samples

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
        for i in range(0,m): 
            norm = np.sqrt(np.sum(d[i,:] * d[i,:]))
            d[i,:] = d[i,:] / norm
        normData = np.column_stack((d,allData[:,-1])) 
        return normData # normalised
    else:
      return allData # unnormalised

'''
    ------------------------------  k-means ------------------------------- 
                                                                             '''
class K_Means():
    
    def __init__(self,dataSet,itr,distMethod):
        self.dataSet = dataSet
        self.itr = itr
        self.distMethod = distMethod

        if self.distMethod == 'distEuclid': # question 1 2 3
            self.distCompute = distEuclid
        elif self.distMethod == 'distManhattan': # question 4 5 
            self.distCompute = distManhattan
        elif self.distMethod == 'distCosine':
            self.distCompute = distCosine
        elif self.distMethod == 'cosineSimi': # question 6
            self.distCompute = cosineSimi

    # k-means main code
    def fit(self,k):   
        
        self.m = np.shape(self.dataSet)[0]  # 329 samples
        self.clusterAssment = np.mat(np.zeros((self.m,2)))
        self.clusterChange = True
        self.oriData = self.dataSet[:,0:-1]
        
        # Stage 1: centroids initialisation
        self.centroids = randCent(self.oriData,k)
        self.a = 0 # iter times counting
        
        while self.clusterChange:
            
            self.clusterChange = False
            
            # do iteration for all samples
            for i in range(self.m):
                
                if self.distMethod != 'cosineSimi':
                    self.minDist = np.inf # loss function Q1-Q5
                else:
                    self.minDist = 0 # for similarity Q6

                self.minIndex = -1

                # do iteration for all centroids
                # Stage 2: find nearest centroid
                if self.distMethod != 'cosineSimi':     # for Q1-Q5

                    for j in range(k):
                        # calculate distance from sample to centroid
                        self.distance = self.distCompute(self.centroids[j,:],self.oriData[i,:])
                        
                        if self.distance < self.minDist: 
                            self.minDist = self.distance
                            self.minIndex = j
                
                else:                                   # for Q6
                    for j in range(k):
                        # calculate similariy from sample to centroid
                        self.similarity = self.distCompute(self.centroids[j,:],self.oriData[i,:])
                        
                        if self.similarity > self.minDist: 
                            self.minDist = self.similarity
                            self.minIndex = j

                # Stage 3: update clusters for each sample
                if self.clusterAssment[i,0] != self.minIndex:
                    self.clusterChange = True # keeping udpate
                    self.clusterAssment[i,:] = self.minIndex,self.minDist**2
            
            # Stage：update centroids
            for j in range(k):
                # get all samples in one cluster
                self.pointsInCluster = self.oriData[np.nonzero(self.clusterAssment[:,0].A == j)[0]]  
                # find avg vector as new centroid  
                if len(self.pointsInCluster!=0):
                    self.centroids[j,:] = np.mean(self.pointsInCluster,axis=0)    
                
            #    self.a += 1 
            # print('update times: ',a)

            # output centroids, clusterAssment
        return self

    def get_sse(self,pointsInCluster,centroid):
        # calculate SSE for a cluster
        return np.sum(np.linalg.norm(pointsInCluster - centroid, 2, 1))

    def evaluate_and_finding_global_optimum_of_kmeans(self,k):
        
        self.max_p, self.max_r, self.max_f,self.purityMax = 0,0,0,0
        self.itr_count=0
        
        for i in range(self.itr):
            
            # k-means run
            self.fit(k)
            self.combination = np.column_stack((self.clusterAssment[:,0],self.dataSet[:,-1]))
            
            # sort result
            self.idex = np.lexsort([self.combination[:, 1].T, self.combination[:, 0].T])
            self.sorted_comb = self.combination[self.idex, :].squeeze()
            
            # cooc-matrix for return TP TN FP FN
            self.cm = get_coocMatrix(self.sorted_comb,k)
            
            # evaluate k-means
            self.p,self.r,self.f = get_p_r_f(self.cm)
            self.purity = get_purity(self.cm)
            self.itr_count+=1
            
            if self.f>self.max_f:
                self.max_p, self.max_r, self.max_f,self.max_purity = self.p,self.r,self.f,self.purity
                self.cm_best=self.cm
            try:
                print("{} times for finding the global optimum, while k equals: {}, keep performing....\n"
                .format(self.itr_count,k))
            except IOError as e:
                print("Don't worry, just face to IO/ Broken Pipe Error, Please rerun the code :D")
        
        if k==4: self.cm_4=self.cm_best # show the best co-oc martix when k=4

        # return max_p, max_r, max_f,purityMax
        return self


    def do_k_1_to_10_test(self):
        # as the name of def, and it will return P,R,F, purity and the best k value

        self.K_List,self.P_List,self.R_List,self.F_List,self.purityList = [],[],[],[],[] # k, precision, recall, f1score
        
        for k in range(1,11):
            # k-means run, return acc
            self.evaluate_and_finding_global_optimum_of_kmeans(k)
            # output
            self.K_List.append(int(k))
            self.P_List.append(round(self.max_p,5))
            self.R_List.append(round(self.max_r,5))
            self.F_List.append(round(self.max_f,5))
            self.purityList.append(round(self.max_purity,5))
            self.k_best = self.K_List[self.F_List.index(max(self.F_List))]
        
        try:
            print("\nAll processes finished! :D\n")
        except IOError as e:
            print("Don't worry, just face to I/O Broken Pipe Error, Please rerun the code :D")
        
        # return K,P,R,F,purityList
        return self

    def visualise(self):
        # plot
        plt.figure(figsize=(8,4)) 
        plt.plot(self.K_List,self.P_List,'b--',linewidth=1,label='precision')  
        plt.plot(self.K_List,self.R_List,'r--',linewidth=1,label='recall') 
        plt.plot(self.K_List,self.F_List,'y--',linewidth=1,label='f1-score') 
        plt.plot(self.K_List,self.purityList,'g--',linewidth=1,label='purity')    
        plt.xlabel("k-clusters")
        plt.title('Accuracy plot, distance measured by: {}'.format(self.distMethod))
        plt.legend(loc='upper right')
        plt.show()

'''
    --------------------------  model evaluation -------------------------- 
                                                                             '''
  
def get_coocMatrix(sorted_comb,k):
    # for calculate TP TN FP FN
    cooc_matrix = [] # generate a zero matrix (2d-list) for future update
    for i in range(c):
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

def get_RI(cooccurrence_matrix):
    # get rand index
    tp,tn,fp,fn = get_tp_tn_fp_fn(cooccurrence_matrix)
    ri = (tp+tn)/(tp+tn+fp+fn)
    return ri

def get_purity(cooccurrence_matrix):
    # calculate purity
    a=0
    for i in cooccurrence_matrix.T:
        a += np.amax(i)
    return a/m

'''
    ---------------------------  math functions ---------------------------- 
                                                                             '''

# random centroids initialisation
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) 
        centroids[i,:] = dataSet[index,:]
    return centroids

# distance measure
def distEuclid(x,y): 
    return np.sqrt(np.sum((x-y)**2))

def distManhattan(x,y):  
    return np.sum(np.abs(x-y))

def distCosine(x,y):
    return 1 - np.dot(x,y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

def cosineSimi(x,y):
    return np.dot(x,y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))

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

'''
    ------------------------------  main part ------------------------------ 
                                                                             '''

if __name__ == '__main__':

    # load, integrate animals, fruits, countries, veggies, with lables
    data = dataGet_Integrate(norm=False)  # QUESTION 1,2,4,6
    question2 = K_Means(dataSet=data,itr=10,distMethod='distEuclid')
    question2.do_k_1_to_10_test() # from k=1 to 10, generate a list of results


    print('list of k:\n',question2.K_List)
    print('\nlist of precision:\n',question2.P_List)
    print('\nlist of recall:\n',question2.R_List)
    print('\nlist of f1-score:\n',question2.F_List)
    print('\nlist of purity:\n',question2.purityList)

    print('\nthe best k is: {}'.format(question2.k_best))
    
    print('\nBest co-oc metrix when k=4:\n',question2.cm_4) # show the best co-oc metrix with highest P,R,F when k=4

    print()
    question2.visualise()
 


