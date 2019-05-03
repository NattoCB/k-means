import k_means as km


if __name__ == '__main__':

    # load, integrate animals, fruits, countries, veggies, with lables
    data = km.dataGet_Integrate(norm=True)  # QUESTION 1,2,4,6
    r1 = km.K_Means(dataSet=data,itr=10,distMethod='cosineSimi')
    r1.do_k_1_to_10_test() # from k=1 to 10, generate a list of results


    print('list of k:\n',r1.K_List)
    print('\nlist of precision:\n',r1.P_List)
    print('\nlist of recall:\n',r1.R_List)
    print('\nlist of f1-score:\n',r1.F_List)
    print('\nlist of purity:\n',r1.purityList)

    print('\nthe best k is: {}'.format(r1.k_best))

    print('\nBest co-oc metrix when k=4:\n',r1.cm_4) # show the best co-oc metrix with highest P,R,F when k=4

    print()
    r1.visualise()