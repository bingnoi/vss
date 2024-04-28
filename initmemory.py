'''
Function:
    Generate the initial memory by using the backbone network
Author:
    Zhenchao Jin
'''
import torch
import warnings
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.models import build_backbone
from mmseg.datasets import build_dataloader, build_dataset
from sklearn.cluster import k_means_
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
warnings.filterwarnings('ignore')

import mmcv
from mmcv.image import tensor2imgs

'''define the cfg'''
cfg = {
    'dataset': {
        'type': 'vspw',
        'set': 'train',
        'rootdir': 'data/vspw/VSPW_480p',
        'aug_opts': [
            ('Resize', {'output_size': (2048, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    },
    'dataloader': {
        'type': 'nondistributed',
        'batch_size': 1,
        'num_workers': 4,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': False,
    },
    'backbone': {
        'type': 'resnet',
        'series': 'resnet',
        'pretrained': True,
        'outstride': 8,
        'use_stem': True,
        'norm_cfg': {'type': 'batchnorm2d', 'opts': {}},
    },
    'memory': {
        'num_feats_per_cls': 1,
        'feats_len': 2048,
        'ignore_index': 255,
        'align_corners': False,
        'savepath': 'init_memory.npy',
        'type': ['random_select', 'clustering'][1],
    }
}


'''cluster by using cosine similarity'''
def cluster(sparse_data, nclust=10):
    def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(X, Y)
    k_means_.euclidean_distances = euc_dist
    scaler = StandardScaler(with_mean=False)
    sparse_data = scaler.fit_transform(sparse_data)
    estimator = k_means_.KMeans(n_clusters=nclust, n_jobs=20)
    _ = estimator.fit(sparse_data)
    return estimator.cluster_centers_


'''main'''
def main():
    # instanced dataset and dataloader
    # dataset_cfg = {'train': cfg['dataset']}
    # dataset = build_dataset(mode='TRAIN', logger_handle=None, dataset_cfg=dataset_cfg)
    # dataloader_cfg = {'train': cfg['dataloader']}
    # dataloader = build_dataloader(mode='TRAIN', dataset=dataset, cfg=dataloader_cfg)
    
    # args = parse_args()
    
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    
    localcfg = mmcv.Config.fromfile("local_configs/mrcfa/B2/mrcfa.b2.480x480.vspw2_hypercorr.160k.py")
    
    dataset = build_dataset(localcfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=localcfg.data.workers_per_gpu,
        dist=True,
        shuffle=False)
    
    # instanced backbone
    backbone_net = build_backbone(localcfg.model.backbone)
    backbone_net = backbone_net.cuda()
    backbone_net.eval()
    # memory cfg
    memory_cfg = cfg['memory']
    assert memory_cfg['type'] in ['random_select', 'clustering']
    # extract feats
    feats_dict = {}
    FloatTensor = torch.cuda.FloatTensor
    pbar = tqdm(enumerate(dataloader))
    for batch_idx, samples in enumerate(dataloader):
        pbar.set_description('Processing %s/%s...' % (batch_idx+1, len(dataloader)))
        # print('keys ',samples.keys())
        # print("img ",samples['img'],samples['gt_semantic_seg'])
        imagetensor, groundtruth = samples['img'].data[0], samples['gt_semantic_seg'].data[0]
        # print(imagetensor,groundtruth)
        
        img_metas = samples['img_metas'].data
        img_metas_norm = img_metas[0][0]['img_norm_cfg']
        
        # print('shape ',imagetensor.shape,groundtruth.shape)
        # image = tensor2imgs(imagetensor, **img_metas_norm)
        # image = image.type(FloatTensor)
        
        feats = backbone_net(imagetensor)[-1]
        gt = groundtruth[0]
        feats = F.interpolate(feats, size=gt.shape, mode='bilinear', align_corners=memory_cfg['align_corners'])
        num_channels = feats.size(1)
        clsids = gt.unique()
        feats = feats.permute(0, 2, 3, 1).contiguous()
        feats = feats.view(-1, num_channels)
        for clsid in clsids:
            clsid = int(clsid.item())
            if clsid == memory_cfg['ignore_index']: continue
            seg_cls = gt.view(-1)
            feats_cls = feats[seg_cls == clsid].mean(0).data.cpu()
            if clsid in feats_dict:
                if (memory_cfg['type'] in ['random_select']) and (len(feats_dict[clsid]) == memory_cfg['num_feats_per_cls']):
                    continue
                feats_dict[clsid].append(feats_cls.unsqueeze(0).numpy())
            else:
                feats_dict[clsid] = [feats_cls.unsqueeze(0).numpy()]
    if memory_cfg['type'] in ['random_select']:
        memory = np.zeros((dataset.num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))
        assert len(feats_dict) == dataset.num_classes
        for idx in range(dataset.num_classes):
            assert len(feats_dict[idx]) == memory_cfg['num_feats_per_cls']
            feats_cls_list = [torch.from_numpy(item) for item in feats_dict[idx]]
            memory[idx] = torch.cat(feats_cls_list, dim=0).numpy()
    elif memory_cfg['type'] in ['clustering']:
        memory = np.zeros((dataset.num_classes, memory_cfg['num_feats_per_cls'], memory_cfg['feats_len']))
        assert len(feats_dict) == dataset.num_classes
        for idx in range(dataset.num_classes):
            feats_cls_list = [torch.from_numpy(item) for item in feats_dict[idx]]
            memory[idx] = cluster(torch.cat(feats_cls_list, dim=0).numpy(), memory_cfg['num_feats_per_cls'])
    np.save(memory_cfg['savepath'], memory)


'''run'''
if __name__ == '__main__':
    with torch.no_grad():
        main()