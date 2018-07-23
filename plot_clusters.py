from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as plt

from sklearn import manifold

from tqdm import tqdm

import pandas as pd
import numpy as np
import os


# DOES: plots the schema_clusters for the csv files. 
def plot_clusters(jacc_matrix, labels, write_path, 
                  overwrite_plot, dataset_name, num_clusters):
 
    plot_mat_path = os.path.join(write_path, 
                                 "plot_" + dataset_name 
                                 + "_k=" + str(num_clusters) + ".npy")
    if not os.path.isfile(plot_mat_path) or overwrite_plot == "1":
        
        # multidimensional scaling to convert distance matrix into 3D
        mds = manifold.MDS(n_components=3, n_jobs=4, 
                           dissimilarity="precomputed", 
                           random_state=1, verbose=2)
        print("Fitting to the distance matrix. ")
        
        # shape (n_components, n_samples)
        pos = mds.fit_transform(jacc_matrix)
        np.save(plot_mat_path,pos)
    else:
        pos = np.load(plot_mat_path)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    print("Setting up plot. ")
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=labels)) 
    
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    for name, group in tqdm(groups):
            
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], 
                       group.y.iloc[t], 
                       group.z.iloc[t], 
                       c=color, marker='o')
            ax.set_aspect('auto')

    plot_3D_path = os.path.join(write_path, "3D_schema_cluster_" 
                                + dataset_name 
                                + "_k=" + str(num_clusters))
    plt.savefig(plot_3D_path, dpi=300)
    print("scatter plot written to \"3D_schema_cluster.png\"")
    return

