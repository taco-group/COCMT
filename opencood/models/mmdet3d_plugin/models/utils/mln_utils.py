import torch
import torch.nn as nn
import math

class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

def transform_object_centers(object_centers, cav2ego, translation=True):
    r""" [MLN Equation 2]:
    transfrom other cavs' predicted 3D object centers
                                            from (other cavs coordinate system) to (ego vehicle coordinate system)
    Args:
        object_centers: other cavs' 3D object centers  -> [B, L-1, Q, 3]
        cav2ego: other cavs' transformation matrix from (other cavs) to (ego vehicle) -> [B, L-1, 4, 4]
    Returns:
        transformed_object_centers: other cavs' 3D object centers at (ego vehicle coordinate system) -> [B, L-1, Q, 3]
    """

    B, L, _, _ = object_centers.size()
    # [B, L-1, Q, 3] -> [B, L-1, Q, 4]
    homogeneous_object_centers = torch.cat([object_centers, torch.ones_like(object_centers[..., 0:1])], dim=-1)
    # 初始化一个张量来保存转换后的坐标
    transformed_object_centers = torch.zeros_like(homogeneous_object_centers)
    # 对每个batch进行操作
    for b in range(B):
        # 对每辆车进行操作
        for i in range(L):
            # 应用转换矩阵到物体中心点
            transformed_object_centers[b, i] = torch.matmul(cav2ego[b, i],
                                                     homogeneous_object_centers[b, i].transpose(0, 1)).transpose(0, 1)
    # 去除齐次坐标的最后一个元素（因为转换后的坐标是齐次坐标）
    transformed_object_centers = transformed_object_centers[..., :3]

    return transformed_object_centers

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb