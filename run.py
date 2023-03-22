import time

from inter_subject_decoder import inter_subject_decoder, permutation_test
from intra_subject_decoder import intra_subject_decoder
from visualization import visualize_decoder, visualize_online, visualize_permutation_test


tic = time.time()

inter_subject_decoder()
visualize_decoder()

# Calculation for intra-subject decoder and confusion matrix
# intra_subject_decoder(save_fn='intra_subject2', tsss_realignment=False)
# visualize_decoder(save_fn='intra_subject2', folder="intra_subject")

# Calculation for inter-subject decoders and confusion matrices
# inter_subject_decoder(mode='sensorspace', save_fn='sensorspace2', tsss_realignment=False)
# visualize_decoder(save_fn='sensorspace2')
# inter_subject_decoder(mode='sensorspace', save_fn='tSSS2', tsss_realignment=True)
# visualize_decoder(save_fn='tSSS2')
# inter_subject_decoder(mode='MCCA', save_fn='MCCA2', tsss_realignment=False)
# visualize_decoder(save_fn='MCCA2')

# from inter_subject_decoder2 import inter_subject_decoder2
# inter_subject_decoder2(save_fn='MCCA_include_all', tsss_realignment=False)
# visualize_decoder(save_fn='MCCA_include_all')

# inter_subject_decoder(n_components_pca=50, n_components_mcca=10, save_fn='MCCA_test', tsss_realignment=False, r=1)
# visualize_decoder(save_fn='MCCA_test')

# Calculation for intra-subject decoder and confusion matrix
# intra_subject_decoder(save_fn='intra_subject', tsss_realignment=False)
# visualize_decoder(save_fn='intra_subject', folder="intra_subject")

# Calculation for inter-subject decoders and confusion matrices
# inter_subject_decoder(mode='sensorspace', save_fn='sensorspace', tsss_realignment=False)
# visualize_decoder(save_fn='sensorspace')
# inter_subject_decoder(mode='sensorspace', save_fn='tSSS', tsss_realignment=True)
# visualize_decoder(save_fn='tSSS')
# inter_subject_decoder(mode='MCCA', save_fn='MCCA_fixed', tsss_realignment=False)
# visualize_decoder(save_fn='MCCA_fixed')

# inter_subject_decoder(mode='PCA', save_fn='PCA', tsss_realignment=False)
# visualize_decoder(save_fn='PCA')

# intra_subject_decoder(save_fn='intra_subject_tsss_pca', tsss_realignment=True, mode='PCA')
# visualize_decoder(save_fn='intra_subject_tsss_pca', folder="intra_subject")
# intra_subject_decoder(save_fn='intra_subject_notsss_pca', tsss_realignment=False, mode='PCA')
# visualize_decoder(save_fn='intra_subject_notsss_pca', folder="intra_subject")
#
# intra_subject_decoder(save_fn='intra_subject_tsss_sensorspace', tsss_realignment=True, mode='sensorspace')
# visualize_decoder(save_fn='intra_subject_tsss_sensorspace', folder="intra_subject")
# intra_subject_decoder(save_fn='intra_subject_notsss_sensorspace', tsss_realignment=False, mode='sensorspace')
# visualize_decoder(save_fn='intra_subject_notsss_sensorspace', folder="intra_subject")

# Calculation and figure for simulated online results, can take a long time (days) to complete
# for i in range(1, 95):
#     inter_subject_decoder(save_fn='online_%dtrials' % i, folder="inter_subject/online", new_subject_trials=i)
#     # visualize_decoder(save_fn='online_%dtrials' % i, folder="inter_subject/online")
# visualize_online()

# Calculation and figure for permutation test, takes a long time (days) to complete
# Reduce number of permutations to improve running time
# Specify number of CPU cores to use for parallel processing with n_jobs (-1 uses all available cores)
# n_permutations = 2000
# permutation_test(n_permutations, start_id=0, save_fn='permutation', n_jobs=-1)
# visualize_permutation_test()

toc = time.time()
print('\nElapsed time: {:.2f} s'.format(toc - tic))
