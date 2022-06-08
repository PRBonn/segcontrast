import numpy as np
import open3d as o3d

def overlap_clusters(cluster_i, cluster_j, min_cluster_point=20):
    # get unique labels from pcd_i and pcd_j
    unique_i = np.unique(cluster_i)
    unique_j = np.unique(cluster_j)

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]

    # also remove clusters with few points
    for cluster in unique_ij.copy():
        ind_i = np.where(cluster_i == cluster)
        ind_j = np.where(cluster_j == cluster)

        if len(ind_i[0]) < min_cluster_point or len(ind_j[0]) < min_cluster_point:
            unique_ij = np.delete(unique_ij, unique_ij == cluster)
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

def clusters_hdbscan(points_set, n_clusters):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusters_from_pcd(pcd, n_clusters):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10))
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, n_clusters):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # segment plane (ground)
    _, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
    pcd_ = pcd.select_by_index(inliers, invert=True)

    labels_ = np.expand_dims(clusters_from_pcd(pcd_, n_clusters), axis=-1)

    # that is a blessing of array handling
    # pcd are an ordered list of points
    # in a list [a, b, c, d, e] if we get the ordered indices [1, 3]
    # we will get [b, d], however if we get ~[1, 3] we will get the opposite indices
    # still ordered, i.e., [a, c, e] which means listing the inliers indices and getting
    # the invert we will get the outliers ordered indices (a sort of indirect indices mapping)
    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    return np.concatenate((points, labels), axis=-1)

def visualize_pcd_clusters(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def visualize_pcd_clusters_compare(point_set, pi, pj):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(point_set[:,:3])

    pi[:,-1], pj[:,-1] = overlap_clusters(pi[:,-1], pj[:,-1])
    point_set[:,-1], pi[:,-1] = overlap_clusters(point_set[:,-1], pi[:,-1])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(pi[:,:3])

    labels = pi[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_i.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_i])
    pcd_i.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_i])

    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(pj[:,:3])

    labels = pj[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_j.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_j])
    pcd_j.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_j])

    # pcd_2 = o3d.geometry.PointCloud()
    # point_set_[:,2] += 10.
    # pcd_2.points = o3d.utility.Vector3dVector(point_set_[:,:3])

    # labels = point_set_[:, -1]
    # import matplotlib.pyplot as plt
    # colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    # colors[labels < 0] = 0

    # pcd_2.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd_i])
    o3d.visualization.draw_geometries([pcd_j])
    #return pcd_
