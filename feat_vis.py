import numpy as np
from utils import *
import MinkowskiEngine as ME
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data_utils.data_map import color_map
import copy
import open3d as o3d
from pcd_utils.pcd_preprocess import clusterize_pcd


def list_segments_points(p_coord, p_feats, labels):
    c_coord = []
    c_feats = []

    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue

            batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)

    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cpu")

    return ME.SparseTensor(
                features=seg_feats,
                coordinates=seg_coord,
                device=device,
            )

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

def find_nearest_neighbors(pcd, points):
    nearest_points = []
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[p], 8)
        nearest_points += list(np.asarray(idx))

    return nearest_points

def visualize_pcd_clusters(point_set, point_set_, point_set__, cmap="viridis", center_point=None):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    if cmap is not None:
        import matplotlib.pyplot as plt
        colors = plt.get_cmap(cmap)((labels - labels.mean()) / labels.std())
    else:
        labels = labels[:, np.newaxis]
        zeros_ = np.zeros_like(labels)
        non_lbl = labels != 1.
        zeros_[non_lbl] = .7
        colors = np.concatenate((labels, zeros_, zeros_), axis=-1)

        idx = find_nearest_neighbors(pcd_, list(np.where(labels == 1.)[0]))
        colors[idx,:] = [1., 0., 0.]

    if center_point is not None:
        colors[center_point,:3] = [1., 0., 0.]
    pcd_.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_2 = o3d.geometry.PointCloud()
    point_set_[:, 2] += 550.
    pcd_2.points = o3d.utility.Vector3dVector(point_set_[:,:3])

    labels = point_set_[:, -1]
    if cmap is not None:
        import matplotlib.pyplot as plt
        colors = plt.get_cmap(cmap)((labels - labels.mean()) / labels.std())
    else:
        labels = labels[:, np.newaxis]
        zeros_ = np.zeros_like(labels)
        non_lbl = labels != 1.
        zeros_[non_lbl] = .7
        colors = np.concatenate((labels, zeros_, zeros_), axis=-1)
        
        idx = find_nearest_neighbors(pcd_2, list(np.where(labels == 1.)[0]))
        colors[idx,:] = [1., 0., 0.]

    if center_point is not None:
        colors[center_point,:3] = [1., 0., 0.]

    pcd_2.colors = o3d.utility.Vector3dVector(colors[:, :3])

    pcd_3 = o3d.geometry.PointCloud()
    point_set__[:, 2] -= 550.
    pcd_3.points = o3d.utility.Vector3dVector(point_set__[:,:3])

    labels = point_set__[:, -1]
    if cmap is not None:
        import matplotlib.pyplot as plt
        colors = plt.get_cmap(cmap)((labels - labels.mean()) / labels.std())
    else:
        labels = labels[:, np.newaxis]
        zeros_ = np.zeros_like(labels)
        non_lbl = labels != 1.
        zeros_[non_lbl] = .7
        colors = np.concatenate((labels, zeros_, zeros_), axis=-1)
        
        idx = find_nearest_neighbors(pcd_3, list(np.where(labels == 1.)[0]))
        colors[idx,:] = [1., 0., 0.]


    if center_point is not None:
        colors[center_point,:3] = [1., 0., 0.]

    pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd_, pcd_2, pcd_3])

def segment_max_pool(feat, seg):
    seg_pool = []
    for s in np.unique(seg):
        print(s, np.argmax(feat[seg == s, 0]))

def global_max_pool(feat):
    return [ np.argmax(feat[:,kern_ind]) for kern_ind in range(feat.shape[-1]) ]

def tanh_pool(feat):
    return list(np.unique(np.where(np.tanh(feat) >= .999)[0]))

def max_pool_vis(feat_d, feat_s):
    print('Global Max Pooled features...')

    c_d, f_d = feat_d.C[:,1:].cpu().detach().numpy(), feat_d.F.cpu().detach().numpy()
    c_s, f_s = feat_s.C[:,1:].cpu().detach().numpy(), feat_s.F.cpu().detach().numpy()

    ind_d = global_max_pool(f_d)

    f_d = np.ones((c_d.shape[0],1))
    f_d = f_d * .7

    f_d[ind_d,0] = 1.
    pcd_d = np.concatenate((c_d, f_d), axis=-1)

    ind_s = global_max_pool(f_s)

    f_s = np.ones((c_s.shape[0],1))
    f_s = f_s * .7

    f_s[ind_s,0] = 1.
    pcd_s = np.concatenate((c_s, f_s), axis=-1)

    visualize_pcd_clusters(pcd_d, pcd_s, None)

def pca_vis(feat_d, feat_s, feat_p, label):
    print('PCA reduced dimension features...')

    print(np.unique(label))
    nonground_ind = label != -1

    c_d, f_d = feat_d.C[:,1:].cpu().detach().numpy(), feat_d.F.cpu().detach().numpy()
    c_s, f_s = feat_s.C[:,1:].cpu().detach().numpy(), feat_s.F.cpu().detach().numpy()
    c_p, f_p = feat_p.C[:,1:].cpu().detach().numpy(), feat_p.F.cpu().detach().numpy()

    pca_d = PCA(n_components=1)
    pca_d.fit(f_d)
    f_d = pca_d.transform(f_d)
    
    pca_s = PCA(n_components=1)
    pca_s.fit(f_s)
    f_s = pca_s.transform(f_s)

    pca_p = PCA(n_components=1)
    pca_p.fit(f_p)
    f_p = pca_p.transform(f_p)

    pcd_d = np.concatenate((c_d, f_d), axis=-1)
    pcd_s = np.concatenate((c_s, f_s), axis=-1)
    pcd_p = np.concatenate((c_p, f_p), axis=-1)
    visualize_pcd_clusters(pcd_d, pcd_s, pcd_p)

def t_sne_vis_(feat):
    print('t-SNE clusters...')

    #colors = [ color_map[int(label)] for label in labels ]
    #colors = np.asarray(colors) / 255.
    #colors = colors[:, ::-1]

    embedded_d = TSNE(n_components=2, perplexity=10.0, init='pca').fit_transform(feat.detach().numpy())
    #embedded_s = TSNE(n_components=2).fit_transform(f_s[ind])

    #fig, ax = plt.subplots(1,2)

    #colors

    # ax[0].scatter(embedded_d[:,0], embedded_d[:,1], c=colors[ind])
    # ax[0].set_title('Depth Contrast')
    plt.scatter(embedded_d[:,0], embedded_d[:,1])#, c=colors[ind])
    #plt.set_title('Segment Contrast')

    plt.show()

def t_sne_vis(feat_d, feat_s, feat_p, x, labels):
    print('t-SNE clusters...')

    labels_ = {
        #0: "unlabeled",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
        }

    # labels_ = {
    #     0: 'unlabeled',
    #     1: 'person',
    #     2: 'rider',
    #     3: 'car',
    #     4: 'trunk',
    #     5: 'plants',
    #     6: 'traffic sign',
    #     7: 'pole',
    #     8: 'trashcan',
    #     9: 'building',
    #     10: 'cone/stone',
    #     11: 'fence',
    #     12: 'bike',
    #     13: 'ground',
    # }

    # color_map = np.array([
    # [0, 0, 0],                       # 0: "unlabeled"
    # [255, 30, 30],                   # 4,5: "1 person"
    # [255, 40, 200],                  # 6: "rider"
    # [100, 150, 245],                 # 7: "car"
    # [135,60,0],                      # 8: "trunk"
    # [0, 175, 0],                     # 9: "plants"
    # [255, 0, 0],                     # 10,11,12: "traffic sign 1"
    # [255, 240, 150],                 # 13: "pole"
    # [125, 255, 0],                   # 14: "trashcan"
    # [255, 200, 0],                   # 15: "building"
    # [50, 255, 255],                  # 16: "cone/stone"
    # [255, 120, 50],                  # 17: "fence"
    # [100, 230, 245],                 # 21: "bike"
    # [128, 128, 128]],                # 22: "ground"
    # dtype = np.uint8)

    c_d, f_d = feat_d.C[:,1:].cpu().detach().numpy(), feat_d.F.cpu().detach().numpy()
    c_s, f_s = feat_s.C[:,1:].cpu().detach().numpy(), feat_s.F.cpu().detach().numpy()
    c_p, f_p = feat_p.C[:,1:].cpu().detach().numpy(), feat_p.F.cpu().detach().numpy()

    sampled_d = []
    sampled_s = []
    sampled_p = []
    colors = []
    pop_legend = []

    for lbl in range(1,20):
        if lbl not in labels_.keys():
            continue
        lbl_ind = np.squeeze(labels == lbl)
        s_d = f_d[lbl_ind,:]
        s_s = f_s[lbl_ind,:]
        s_p = f_p[lbl_ind,:]

        if len(s_s) == 0:
            continue

        ind = np.random.choice(len(s_s), min(len(s_s),200), replace=False)

        sampled_d.append(s_d[ind])
        sampled_s.append(s_s[ind])
        sampled_p.append(s_p[ind])
        colors.append([ np.asarray(color_map[lbl][::-1]) / 255. for _ in range(len(ind))])

        pop_legend.append(mpatches.Patch(color=np.asarray(color_map[lbl][::-1]) / 255., label=labels_[lbl]))


    f_d = np.vstack(sampled_d)
    f_s = np.vstack(sampled_s)
    f_p = np.vstack(sampled_p)
    colors = np.vstack(colors)

    colors_ = [ np.asarray(color_map[int(label)][::-1]) / 255. for label in y ]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(c_d)
    pcd.colors = o3d.utility.Vector3dVector(colors_)
    o3d.visualization.draw_geometries([pcd])

    # for perp in [200,10,40,50,80,90]:
    #     for lr in [100]:
    #         #perp 70 - 80 worked better
    #print('Perp: ', perp, ' lr: ', lr)
    print(int(np.sqrt(len(f_d))/2))
    embedded_d = TSNE(n_components=2, perplexity=int(np.sqrt(len(f_d))/2), learning_rate=100, n_iter=50000, init='pca', n_jobs=4).fit_transform(f_d)
    embedded_s = TSNE(n_components=2, perplexity=int(np.sqrt(len(f_d))/2), learning_rate=100, n_iter=50000, init='pca', n_jobs=4).fit_transform(f_s)
    embedded_p = TSNE(n_components=2, perplexity=int(np.sqrt(len(f_d))/2), learning_rate=100, n_iter=50000, init='pca', n_jobs=4).fit_transform(f_p)

    fig, ax = plt.subplots(1,3)

    ax[0].scatter(embedded_d[:,0], embedded_d[:,1], c=colors)
    ax[0].set_title('Depth Contrast')
    ax[1].scatter(embedded_p[:,0], embedded_p[:,1], c=colors)
    ax[1].set_title('Point Contrast')
    ax[2].scatter(embedded_s[:,0], embedded_s[:,1], c=colors)
    ax[2].set_title('Segment Contrast')

    plt.legend(handles=pop_legend)
    plt.show()
    

def individual_feats(feat_d, feat_s):
    print('Individual kernel features...')

    c_d, f_d = feat_d.C[:,1:].cpu().detach().numpy(), feat_d.F.cpu().detach().numpy()
    c_s, f_s = feat_s.C[:,1:].cpu().detach().numpy(), feat_s.F.cpu().detach().numpy()

    for i in range(f_d.shape[-1]):
        pcd_d = np.concatenate((c_d, np.tanh(f_d[:,i,np.newaxis])), axis=-1)
        pcd_s = np.concatenate((c_s, np.tanh(f_s[:,i,np.newaxis])), axis=-1)
        visualize_pcd_clusters(pcd_d, pcd_s)

def saliency(x, y, model_d, model_s, model_p, head_d, head_s):
    print('Saliency maps over global max pooled features...')

    labels = {
        #0: "unlabeled",
        1: "car",
        2: "bicycle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        18: "pole",
        19: "traffic-sign",
    }

    for param in model_d.parameters():
        param.require_grads = False

    for param in model_s.parameters():
        param.require_grads = False

    for param in head_d.parameters():
        param.require_grads = False

    for param in head_s.parameters():
        param.require_grads = False

    c_d = x.C[:,1:].cpu().detach().numpy()

    model_d.eval()
    model_s.eval()
    head_d.eval()
    head_s.eval()

    x.F.requires_grad = True

    for class_ in labels.keys():
        class_ind = y.cpu().detach().numpy()
        class_ind = class_ind == class_

        if class_ == 0 or not class_ind.any():
            continue

        rand_cls_point = np.where(class_ind == True)[0]
        rand_cls_point = rand_cls_point[np.random.choice(len(rand_cls_point))]

        print(f'Saliency for {labels[class_]}')

        x.F.grad = None

        out_d = model_d(x)
        #outh_d = head_d(out_d)

        #out_d = torch.unsqueeze(torch.mean(out_d.F, dim=-1), dim=-1)
        # pred = out_d.max(dim=1)[1] == class_
        # class_ind_ = torch.nn.functional.one_hot(out_d.max(dim=1)[1], num_classes=20)
        # out_d = out_d[class_ind_.bool()]#out_d[:,class_]

        #score_d = torch.mean(out_d[pred], dim=-1)
        score_d = torch.mean(out_d.F[rand_cls_point], dim=-1)
        
        x.F.retain_grad()
        score_d.backward()
        
        slc_d, _ = torch.max(torch.abs(x.F.grad), dim=-1)
        slc_d = (slc_d - slc_d.min())/(slc_d.max()-slc_d.min())
        slc_d = slc_d.cpu().detach().numpy()
        pcd_d = np.concatenate((c_d, slc_d[:, np.newaxis]), axis=-1)

        x.F.grad = None

        out_s = model_s(x)
        #outh_s = head_s(out_s)

        #out_s = torch.unsqueeze(torch.mean(out_s.F, dim=-1), dim=-1)
        # pred = out_s.max(dim=1)[1] == class_
        # class_ind_ = torch.nn.functional.one_hot(out_s.max(dim=1)[1], num_classes=20)
        # out_s = out_s[class_ind_.bool()]#out_s[:,class_]
        
        # score_s = torch.mean(out_s[pred], dim=-1)
        score_s = torch.mean(out_s.F[rand_cls_point], dim=-1)

        x.F.retain_grad()
        score_s.backward()

        slc_s, _ = torch.max(torch.abs(x.F.grad), dim=-1)
        slc_s = (slc_s - slc_s.min())/(slc_s.max()-slc_s.min())
        slc_s = slc_s.cpu().detach().numpy()

        pcd_s = np.concatenate((c_d, slc_s[:, np.newaxis]), axis=-1)

        x.F.grad = None

        out_p = model_p(x)
        #outh_s = head_s(out_p)

        #out_p = torch.unsqueeze(torch.mean(out_p.F, dim=-1), dim=-1)
        # pred = out_p.max(dim=1)[1] == class_
        # class_ind_ = torch.nn.functional.one_hot(out_p.max(dim=1)[1], num_classes=20)
        # out_p = out_p[class_ind_.bool()]#out_p[:,class_]
        
        # score_s = torch.mean(out_p[pred], dim=-1)
        score_p = torch.mean(out_p.F[rand_cls_point], dim=-1)

        x.F.retain_grad()
        score_p.backward()

        slc_p, _ = torch.max(torch.abs(x.F.grad), dim=-1)
        slc_p = (slc_p - slc_p.min())/(slc_p.max()-slc_p.min())
        slc_p = slc_p.cpu().detach().numpy()

        pcd_p = np.concatenate((c_d, slc_p[:, np.newaxis]), axis=-1)

        visualize_pcd_clusters(pcd_d, pcd_s, pcd_p, center_point=rand_cls_point)
        #torch.cuda.empty_cache()

if __name__ == '__main__':
    device = torch.device("cpu")

    model_d = sparse_models['MinkUNetSMLP'](
            in_channels=4,
            out_channels=latent_features['MinkUNet'],
        ).type(torch.FloatTensor)
    #model_d.to('cuda')

    model_s = sparse_models['MinkUNet'](
            in_channels=4,
            out_channels=latent_features['MinkUNet'],
        ).type(torch.FloatTensor)
    #model_s.to('cuda')

    model_p = sparse_models['MinkUNet'](
            in_channels=4,
            out_channels=latent_features['MinkUNet'],
        ).type(torch.FloatTensor)
    #model_p.to('cuda')

    head_d = SegmentationClassifierHead(in_channels=96, out_channels=20).type(torch.FloatTensor)
    head_s = SegmentationClassifierHead(in_channels=96, out_channels=20).type(torch.FloatTensor)
    #head_d.to('cuda')
    #head_s.to('cuda')

    model_d.eval()
    model_s.eval()
    model_p.eval()
    head_d.eval()
    head_s.eval()

    checkpoint_d = torch.load(f'checkpoint/checkpoint-ep200.pth.tar', map_location=torch.device('cpu'))
    model_d.load_state_dict(checkpoint_d['vox'])

    checkpoint_s = torch.load(f'checkpoint/lastepoch199_model_segment_contrast.pt', map_location=torch.device('cpu'))
    model_s.load_state_dict(checkpoint_s['model'])

    checkpoint_p = torch.load(f'checkpoint/checkpoint_214200.pth', map_location=torch.device('cpu'))
    model_p.load_state_dict(checkpoint_p['state_dict'])

    # checkpoint_head = torch.load(f'checkpoint/lastepoch14_model_head_depth_contrast_0p5.pt', map_location=torch.device('cuda'))
    # head_d.load_state_dict(checkpoint_head['model'])

    # checkpoint_head = torch.load(f'checkpoint/lastepoch14_model_head_segment_contrast_segonly_0p5.pt', map_location=torch.device('cuda'))
    # head_s.load_state_dict(checkpoint_head['model'])


    #data_val = data_loaders['SemanticPOSS'](root='/home/lucas/Downloads/SemanticPOSS_dataset', split='validation', intensity_channel=True, pre_training=False, resolution=0.05)
    data_val = data_loaders['SemanticKITTI'](root='./Datasets/SemanticKITTI', split='validation', intensity_channel=True, pre_training=False, resolution=0.05)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=1, collate_fn=SparseCollation(0.05, np.inf), shuffle=True)

    data_iterator = iter(val_loader)

    counter = 0

    feat_accum = None

    while True:
        x_coord, x_feats, x_label = data_iterator.next()

        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        feat_d = model_d(x)
        feat_s = model_s(x)
        feat_p = model_p(x)

        # points_set = feat_s.C.detach().numpy()[:,1:]
        # points_seg = np.expand_dims(clusterize_pcd(points_set * 0.05, 50)[:,-1], axis=0)
        # feat_seg = list_segments_points(feat_s.C, feat_s.F, points_seg)
        # feat = head(feat_seg)
        # feat = torch.nn.functional.normalize(feat, dim=1)
        # #seg_c, seg_f = feat_seg.decomposed_coordinates_and_features
        # feat_accum = feat if feat_accum == None else torch.cat((feat_accum, feat), dim=0)
        # print(feat_accum.shape)
        # counter += 1
        #max_pool_vis(feat_d, feat_s)
        saliency(x, y, model_d, model_s, model_p, head_d, head_s)
        #pca_vis(feat_d,feat_s, feat_p, y)
        torch.cuda.empty_cache()
        #individual_feats(feat_d, feat_s)
    
        #t_sne_vis(feat_d, feat_s, feat_p, x, y)
