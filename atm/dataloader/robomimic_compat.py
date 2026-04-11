"""
Minimal vendored subset of robomimic used by ATM's track dataloaders.

Adapted from:
- third_party/robomimic/robomimic/utils/obs_utils.py
- third_party/robomimic/robomimic/utils/tensor_utils.py
- third_party/robomimic/robomimic/models/obs_core.py
"""

import collections

import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to nested dictionaries, lists, or tuples.
    """
    assert list not in type_func_dict
    assert tuple not in type_func_dict
    assert dict not in type_func_dict

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    if isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret

    for t, f in type_func_dict.items():
        if isinstance(x, t):
            return f(x)
    raise NotImplementedError(f"Cannot handle data type {type(x)}")


def map_tensor(x, func):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda value: value,
        },
    )


def unsqueeze(x, dim):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda value: value.unsqueeze(dim=dim),
            np.ndarray: lambda value: np.expand_dims(value, axis=dim),
            type(None): lambda value: value,
        },
    )


def flatten_single(x, begin_axis=1):
    fixed_size = x.size()[:begin_axis]
    new_shape = list(fixed_size) + [-1]
    return x.reshape(*new_shape)


def flatten(x, begin_axis=1):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda value, b=begin_axis: flatten_single(value, begin_axis=b),
        },
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    assert begin_axis <= end_axis
    assert begin_axis >= 0
    assert end_axis < len(x.shape)
    assert isinstance(target_dims, (tuple, list))

    shape = x.shape
    final_shape = []
    for i in range(len(shape)):
        if i == begin_axis:
            final_shape.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_shape.append(shape[i])
    return x.reshape(*final_shape)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda value, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                value, begin_axis=b, end_axis=e, target_dims=t
            ),
            np.ndarray: lambda value, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                value, begin_axis=b, end_axis=e, target_dims=t
            ),
            type(None): lambda value: value,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda value, b=begin_axis, e=end_axis: reshape_dimensions_single(
                value, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            np.ndarray: lambda value, b=begin_axis, e=end_axis: reshape_dimensions_single(
                value, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            type(None): lambda value: value,
        },
    )


def expand_at_single(x, size, dim):
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    return map_tensor(x, lambda tensor, s=size, d=dim: expand_at_single(tensor, s, d))


def unsqueeze_expand_at(x, size, dim):
    x = unsqueeze(x, dim)
    return expand_at(x, size, dim)


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    Crop images using top-left crop indices.
    """
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    crop_in_grid = torch.cat((crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1)

    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)
    all_crop_inds = all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    all_crop_inds = unsqueeze_expand_at(all_crop_inds, size=image_c, dim=-3)
    all_crop_inds = flatten(all_crop_inds, begin_axis=-2)

    images_to_crop = unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    reshape_axis = len(crops.shape) - 1
    crops = reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    """
    Randomly sample crops from images, matching robomimic's implementation.
    """
    device = images.device

    source_im = images
    if pos_enc:
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))

        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)
        source_im = torch.cat((source_im, position_enc), dim=-3)

    _, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1)

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )
    return crops, crop_inds


class CropRandomizer:
    """
    Minimal copy of robomimic CropRandomizer for ATM dataloader augmentation.
    """

    def __init__(
        self,
        input_shape,
        crop_height=76,
        crop_width=76,
        num_crops=1,
        pos_enc=False,
    ):
        assert len(input_shape) == 3
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        return list(input_shape)

    def _forward_in(self, inputs):
        assert len(inputs.shape) >= 3
        out, _ = sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        return join_dimensions(out, 0, 1)
