## k-means Algorithm ## 

                            																Author: Jasper Fang

                            																Python version: 3.0
                            																Matplotlib version: 3.0.3
                            																Numpy version: 1.16.2
                            																IDE: Sublime 3



    ### if the code cannot be run, please follow the steps at the end of this document ###


# Data Loading:
   
   * get original data with labels
	 
	 At the very beginning, the project will load 4 types of data respectively, which are animals(50),
	 
	 fruits(58), veggies(60), and countries(161), there are 329 samples in total. In which each sample
	 
	 has 300 features (from word embedding). Moreover, `get_data(dataPath,label)` function will also label 
	 
	 each group of samples from 1 to 4, which as the 301st feature of the data, and return them finally.

   * integrate (and nomarlise) data
     
     `dataIntegrate(norm=False)` function will call `get_data(dataPath,label)` 4 times for loading all 
     
     the datasets, and integrate all 329 data together (by row_stacking of the array). Furthermore, this 
     
     function can be used to decide nomarlise data or not, by controlling `norm` parameter as True or False.


# Math functions:
    
   There have some functions for future use:

   * Distance measure
     
     used by K-Means algorithm, for calculating the distance of 2 input vectors. In this project, it 

     can do 4 types of distance measurements, which are `Euclidean distance`, `Manhattan distance`, 
  
     `Cosine distance`, and `Cosine Similarity` respectively. 

   * Factorial
 	 
 	   used by Combination calculations, return the factorial of input integer

   * Combination

     used by co-occurrence matrix (for calculating TP,TN,FP,FN), return C(n r) for input number or array


# K-Means:
   
   * main class

   	 The algorithm will receive a dataset, specified k-number, and measurement method of distance firstly,
    
     then generate a set of random centroids for initialisation, and set instances from the dataset randomly.
    
     After that, the algorithm will assign all other instances to the closest cluster centroids and calculate
    
     the mean of each cluster, repeat this step until there is no change of clusters. 
    
     Finally return all centroids and clusterAssments.

	
# Model test and run:
   
   * do_k_1_to_10_test(data,distMethod) function

     can iter 10 times to create(record) a list of results, for visualise and further comparsion

   * find_highest_acc_of_kmeans(data,distMethod,k) function

     can iter 5 times (in default) to find a highest accuracy of current algorithm with k number of clusters

     in order to prevent the disadvantage of k-means algorithm that may converge into local optimum


# Model Evaluation:
   
   * co-oc matrix

     k-means will output an array contains 2 columns: [clusters, origin class], and we sum this pair of 

     output data into a co-oc matrix like the table below:

     +-----+-----+-----+-----+----
     |     |  k1 |  k2 |  k3 |...
     +-----+-----+-----+-----+----
     |  c1 |  5  |  1  |  67 |...
     +-----+-----+-----+-----+----
     |  c2 |  76 |  3  |  13 |...     
     +-----+-----+-----+-----+----
     | ... |  2  |  93 |  7  |...
     +-----+-----+-----+-----+----

   * TP,TN,FP,FN

     In k-means algorithm, (like the previous co-oc matrix table, when k=3 c=3) 
	     
	     TP+FP = C(83 2) + C(97 2) + C(87 2)
	     TP+FN = C(83 2) + C(92 2) + C(102 2)
	     TP+TN+FP+FN = C(n 2) = C(267 2)
	     TP = C(5 2) + C(76 2) + C(2 2) + C(3 2) + C(93 2) + C(67 2) + C(13 2) + C(7 2)
	     FP = (TP+FP) - TP
	     FN = (TP+FN) - TP
	     TN = (TP+TN+FP+FN) - TP - FP - FN

     And `get_tp_tn_fp_fn(co-ocMetrix)` will calculate these 4 measurements follows above method and return.

   * Precision, Recall, F1Score

     `get_p_r_f(co-ocMetrix)` function will use TP,TN,FP,FN to calculate Precision, Recall, and F1score.

     follows the formula: P=TP/(TP+FP), R=TP/(TP+FN), F1=(2*P*R)/(P+R)

   * Rand Index
   	 
   	 `get_RI(co-ocMetrix)`  will calculate to Rand Index of k-means' result, which follows the formula:

   	 RI = (TP+TN)/(TP+TN+FP+FN)

   * Purity

   	 `get_purity(co-ocMetrix)` receives co-oc matrix, and will find the most frequent class of every cluster

   	 them sum together to return the Purity of k-means' result.

   * k_best

     After all results are generated, the best k value will be found from the highest F1-score's result.


# Visualisation:
   
   Finally, the `visualise()` function will receives the final list of results and visualise them.


# Test

```python
    
    import k_means as km

    data = km.dataGet_Integrate(norm=False)  # option for normalisation or not

    aVeryCuteKmeans = km.K_Means(dataSet=data,itr=10,distMethod='distEuclid') # option for 3 distance measurments
    # or use 'distEuclid'
    # or use 'distManhattan'
    # or use 'cosineSimi'
    # or     'distCosine'

    aVeryCuteKmeans.do_k_1_to_10_test() # from k=1 to 10, calculate and generate a list of results

    aVeryCuteKmeans.visualise()

```

                  and the output parameters as below:

```python

    aVeryCuteKmeans.K_List # k from 1 to 10
    aVeryCuteKmeans.P_List # precision list from k=1 to 10
    aVeryCuteKmeans.R_List # recall list from k=1 to 10
    aVeryCuteKmeans.F_List # f1-score list from k=1 to 10
    aVeryCuteKmeans.purityList # purity list from k=1 to 10
    aVeryCuteKmeans.k_best # the best k of all results
    aVeryCuteKmeans.cm_4 # the best result (co-oc metrix) with the highest F1-Score and accuracy when k=4

```




## If the code cannot be run successfully ##

1. BrokenPipeError
	delete print() function and replaced by an output Stream or run in Anaconda or Colab

2. ERROR at `import matplotlib.pyplot as plt`
	add a code before import plt, `mpl.use('TkAgg')` for changing the backend of plt 3.0.3 in macOS

3. Unknown Error
	contact the author tofangsiyu@gmail.com



