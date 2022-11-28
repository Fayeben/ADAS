import torch
from functools import partial


def collate_scn_base(input_dict_list, output_orig, output_image=True):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]
    points_img=[]
    camera_path=[]
    points=[]
    lidar_path=[]
    boxes=[]
    sample_token=[]
    scene_name=[]
    calib=[]

    if output_image:
        imgs = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        orig_points_idx = []

    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []

    for idx, input_dict in enumerate(input_dict_list):
        # print(input_dict.keys())
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        points_img.append(input_dict['points_img'])
        camera_path.append(input_dict['camera_path'])
        points.append(input_dict['points'])
        lidar_path.append(input_dict['lidar_path'])
        boxes.append(input_dict['boxes'])
        sample_token.append(input_dict['sample_token'])
        scene_name.append(input_dict['scene_name'])
        calib.append(input_dict['calib'])

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    out_dict["points_img"] = points_img
    out_dict["camera_path"] = camera_path
    out_dict["points"] = points
    out_dict["points_img"] = points_img
    out_dict["lidar_path"] = lidar_path
    out_dict["boxes"] = boxes
    out_dict["sample_token"] = sample_token
    out_dict["scene_name"] = scene_name
    out_dict["calib"] = calib
    return out_dict


def get_collate_scn(is_train):
    return partial(collate_scn_base,
                   output_orig=not is_train,
                   )
