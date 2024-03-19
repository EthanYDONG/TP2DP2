import numpy as np
from collections import Counter

def purity(predict_index, real_index):
    real_num = real_index.max() + 1
    predict_num = predict_index.max() + 1
    success_cluster = 0
    for i in range(predict_num):
        tmp_count = np.zeros(real_num)
        for j in range(real_index.shape[0]):
            if (predict_index[j] == i):
                tmp_count[real_index[j]] += 1
        success_cluster += tmp_count.max()
    return success_cluster / predict_index.shape[0]   

def calculate_purity(clusters, labels):

    total_samples = len(clusters)
    
    cluster_label_counts = {}
    for cluster, label in zip(clusters, labels):
        if cluster not in cluster_label_counts:
            cluster_label_counts[cluster] = Counter()
        cluster_label_counts[cluster][label] += 1

    
    majority_counts = [counter.most_common(1)[0][1] for counter in cluster_label_counts.values()]

    
    purity = sum(majority_counts) / total_samples
    return purity
