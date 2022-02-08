import numpy as np
import matplotlib.pyplot as plt
import prepare_data
from MCCA import obtainCCA, testCCA, transformTrials

def test_intersubject_corr(KK=50, K=20):
    """
    Test if the inter-subject correlations/consistency observed from training
    data CCAs generalize over testing data.
    
    Parameters:
        KK (int): Number of PCAs to retain for each subject (default 50)
        K (int): Number of CCAs to retain (default 20)
    """
    data_train, data_test = prepare_data.load_zhang_data()
    # apply M-CCA (training) to half of the data
    W,mu,sigma,weights = obtainCCA(data_train,KK,K)
    
    # testing on left-out half of the data
    correlations = testCCA(data_train,data_test,W,mu,sigma,weights)
    
    # visualize the CCA performance (inter-subject correlations over testing data)
    plt.plot(correlations)
    plt.legend(('Training','Testing'))
    plt.xticks(range(0,K+1,5))
    plt.xlabel('CCA components')
    plt.ylabel('Averaged inter-subject correlations')
    plt.savefig(prepare_data.results_folder+'CCA_components_K'+str(K)+'_KK'
                +str(KK)+'.png',format='png')
    plt.show()
    

def test_regularization(KK=50, K=20, r=[0, 10, 20, 50, 100, 200, 500]):
    """ 
    Visualize the CCA performance (inter-subject correlations over testing data
    averaged over the first 10 CCA).                
    This plot demonstrates that obtained CCAs generalizes well to testing data up
    to lambda = 100; one would need to combine this information with
    the inter-subject classification performance and the CCA weight maps to 
    decide the best value of regularization.
        
    Parameters:
        KK (int): Number of PCAs to retain for each subject (default 50)
        K (int): Number of CCAs to retain (default 20)
        r (list of int or float): Regularization values to test
    """
    data_train, data_test = prepare_data.load_zhang_data()
    
    correlations = np.zeros((len(r),K,2))
    for i in range(len(r)):
        # apply M-CCA (training) to half of the data
        W,mu,sigma,weights = obtainCCA(data_train,KK,K,True,r[i])
        # testing on left-out half of the data (testing)
        correlations[i] = testCCA(data_train,data_test,W,mu,sigma,weights)

    plt.plot(np.squeeze(np.mean(correlations[:,0:10],1)))
    plt.legend(('Training','Testing'))
    plt.xticks(range(len(r)),labels=[str(item) for item in r])
    plt.xlabel(r'Regularization ($\lambda$)')
    plt.ylabel('Averaged inter-subject correlations')
    plt.savefig(prepare_data.results_folder+'regularization.png',format='png')
    plt.show()
    
def test_number_of_PCAs(K=20):
    """  
    Visualize the CCA performance for a fixed number of CCA components and 
    varying number of PCA components.
        
    Parameters:
        K (int): Number of CCAs to retain (default 20)
    """
    data_train, data_test = prepare_data.load_zhang_data()
    n_channels = data_train.shape[2]
    KK = np.arange(K,n_channels)
    correlations = np.zeros((len(KK),K,2))
    for i in range(len(KK)):
        # apply M-CCA (training) to half of the data
        W,mu,sigma,weights = obtainCCA(data_train,KK[i],K)
        # testing on left-out half of the data
        correlations[i] = testCCA(data_train,data_test,W,mu,sigma,weights)
    correlations = np.stack(correlations)
    # visualize the MCCA performance for different number of PCAs
    plt.plot(KK,np.squeeze(np.mean(correlations,1)))
    plt.legend(('Training','Testing'))
    #plt.xticks(range(len(KK)),labels=[str(item) for item in KK])
    plt.xlabel('PCA components')
    plt.ylabel('Averaged inter-subject correlations')
    plt.savefig(prepare_data.results_folder+'PCA_components_K'+str(K)+'.png',format='png')
    plt.show()
    
def test_single_trial_transform(KK=50,K=20):
    """ 
    Use averaged trials to calculate M-CCA transformation weights and apply
    them to single trial data to transform it into the shared M-CCA space.
    
    Parameters:
        KK (int): Number of PCAs to retain for each subject (default 50)
        K (int): Number of CCAs to retain (default 20)
    Returns:
        CCAs (ndarray): Single trial data in M-CCA space (trials, samples, CCAs)
    """
    data_averaged, data_st = prepare_data.load_zhang_data_st()
    # apply M-CCA to averaged data
    W,mu,sigma,weights = obtainCCA(data_averaged,KK,K)
    # transform single-trial data from sensor dimensions to CCA
    CCAs = transformTrials(data_st,W,mu,sigma,weights)
    # dimensions in 'CCAs' are assumed to align from subject to subject
    return CCAs

if __name__=="__main__":
    import time
    tic = time.time()
    
    test_intersubject_corr()
    test_regularization()
    #test_number_of_PCAs()
    CCAs = test_single_trial_transform()

    toc = time.time()
    print('\nElapsed time: {:.2f} s'.format(toc - tic))