import torch
import torch.nn as nn
# from models.encoders.DFormer import LayerNorm
# from mmseg.models.decode_heads.mask_attention_cot import LayerNorm as LayerNorm1


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif (
            isinstance(m, norm_layer)
            or isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
            or isinstance(m, nn.GroupNorm)
            or isinstance(m, nn.LayerNorm)
            # or isinstance(m, LayerNorm)
        ):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    # assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    print(
        "Weight Decay:",
        len(group_decay),
        "Weight No Decay:",
        len(group_no_decay),
        "Total:",
        len(list(module.parameters())),
    )
    # for i in list(module.parameters()):
    #     print(type(i), i.size())
    # assert 1==2
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=0.0, lr=lr))
    return weight_group


def build_e1_optimizer_param_groups(
    module,
    norm_layer,
    *,
    base_lr,
    new_lr,
    weight_decay,
    new_parameter_prefixes=("feature_adapter.",),
    expected_geo_weight_count=29,
):
    """Build the frozen four-group optimizer identity for MMFR E1 Batch 1A.

    Existing Conv/Linear and normalization classification follows ``group_weight``.
    The only permitted previously-unmanaged base parameters are the audited
    ``*.Geo.weight`` tensors, which are added to ``base_decay``. Every trainable
    parameter must appear in exactly one named group.
    """

    trainable = {name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad}
    decay_ids = set()
    no_decay_ids = set()
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    for submodule in module.modules():
        if isinstance(submodule, nn.Linear) or isinstance(submodule, conv_types):
            if submodule.weight is not None and submodule.weight.requires_grad:
                decay_ids.add(id(submodule.weight))
            if submodule.bias is not None and submodule.bias.requires_grad:
                no_decay_ids.add(id(submodule.bias))
        elif isinstance(submodule, norm_types) or isinstance(submodule, norm_layer):
            if getattr(submodule, "weight", None) is not None and submodule.weight.requires_grad:
                no_decay_ids.add(id(submodule.weight))
            if getattr(submodule, "bias", None) is not None and submodule.bias.requires_grad:
                no_decay_ids.add(id(submodule.bias))

    overlap = decay_ids & no_decay_ids
    if overlap:
        raise ValueError(f"E1 optimizer classification overlap for {len(overlap)} parameters")

    classified = decay_ids | no_decay_ids
    unmanaged = [name for name, parameter in trainable.items() if id(parameter) not in classified]
    geo_names = sorted(name for name in unmanaged if name.endswith(".Geo.weight"))
    unexpected = sorted(set(unmanaged) - set(geo_names))
    if unexpected:
        raise ValueError(f"E1 optimizer found unexpected unmanaged parameters: {unexpected}")
    if len(geo_names) != int(expected_geo_weight_count):
        raise ValueError(
            f"E1 optimizer expected {expected_geo_weight_count} Geo.weight tensors, got {len(geo_names)}"
        )
    for name in geo_names:
        decay_ids.add(id(trainable[name]))

    prefix_tuple = tuple(str(prefix) for prefix in new_parameter_prefixes)
    buckets = {
        "base_decay": [],
        "base_no_decay": [],
        "new_decay": [],
        "new_no_decay": [],
    }
    names_by_group = {name: [] for name in buckets}
    membership = {name: 0 for name in trainable}
    for name, parameter in trainable.items():
        is_new = name.startswith(prefix_tuple)
        parameter_id = id(parameter)
        if parameter_id in decay_ids:
            group_name = "new_decay" if is_new else "base_decay"
        elif parameter_id in no_decay_ids:
            group_name = "new_no_decay" if is_new else "base_no_decay"
        else:
            raise ValueError(f"E1 optimizer failed to classify trainable parameter {name}")
        buckets[group_name].append(parameter)
        names_by_group[group_name].append(name)
        membership[name] += 1

    invalid_membership = {name: count for name, count in membership.items() if count != 1}
    if invalid_membership:
        raise ValueError(f"E1 optimizer membership must equal one: {invalid_membership}")

    lr_scale = float(new_lr) / float(base_lr)
    groups = [
        {
            "group_name": "base_decay",
            "params": buckets["base_decay"],
            "lr": float(base_lr),
            "lr_scale": 1.0,
            "weight_decay": float(weight_decay),
        },
        {
            "group_name": "base_no_decay",
            "params": buckets["base_no_decay"],
            "lr": float(base_lr),
            "lr_scale": 1.0,
            "weight_decay": 0.0,
        },
        {
            "group_name": "new_decay",
            "params": buckets["new_decay"],
            "lr": float(new_lr),
            "lr_scale": lr_scale,
            "weight_decay": float(weight_decay),
        },
        {
            "group_name": "new_no_decay",
            "params": buckets["new_no_decay"],
            "lr": float(new_lr),
            "lr_scale": lr_scale,
            "weight_decay": 0.0,
        },
    ]
    for group in groups:
        group["parameter_names"] = tuple(sorted(names_by_group[group["group_name"]]))
    return groups


def configure_optimizers(model, lr, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        torch.nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.Embedding,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        LayerNorm,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif "layer_scale" in fpn:
                # special case layer norm scaling parameters in the T5 model
                decay.add(fpn)
            elif "depth_scale" in fpn:
                # special case depth norm scaling parameters in the T5 model
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    # no_decay.add("pos_emb")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, (
        "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return optim_groups
