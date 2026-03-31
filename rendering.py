import torch
import torch.nn.functional as F
from . import wp, HAS_WARP

def contract_to_unit_cube(pts, aabb_min, aabb_max):
    return (pts - aabb_min) / (aabb_max - aabb_min)


if HAS_WARP:
    @wp.kernel
    def ray_aabb_intersection_kernel(
        origins: wp.array(dtype=wp.float32, ndim=2),
        dirs: wp.array(dtype=wp.float32, ndim=2),
        aabb_min_x: float, aabb_min_y: float, aabb_min_z: float,
        aabb_max_x: float, aabb_max_y: float, aabb_max_z: float,
        t_near: wp.array(dtype=wp.float32, ndim=1),
        t_far: wp.array(dtype=wp.float32, ndim=1),
        valid_mask: wp.array(dtype=wp.int8, ndim=1)
    ):
        tid = wp.tid()
        
        ox = origins[tid, 0]
        oy = origins[tid, 1]
        oz = origins[tid, 2]
        
        # Add small epsilon to prevent division by zero
        dx = dirs[tid, 0] + 1e-8
        dy = dirs[tid, 1] + 1e-8
        dz = dirs[tid, 2] + 1e-8
        
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dz = 1.0 / dz
        
        t0_x = (aabb_min_x - ox) * inv_dx
        t1_x = (aabb_max_x - ox) * inv_dx
        t_min_x = wp.min(t0_x, t1_x)
        t_max_x = wp.max(t0_x, t1_x)
        
        t0_y = (aabb_min_y - oy) * inv_dy
        t1_y = (aabb_max_y - oy) * inv_dy
        t_min_y = wp.min(t0_y, t1_y)
        t_max_y = wp.max(t0_y, t1_y)
        
        t0_z = (aabb_min_z - oz) * inv_dz
        t1_z = (aabb_max_z - oz) * inv_dz
        t_min_z = wp.min(t0_z, t1_z)
        t_max_z = wp.max(t0_z, t1_z)
        
        near = wp.max(wp.max(t_min_x, t_min_y), t_min_z)
        far = wp.min(wp.min(t_max_x, t_max_y), t_max_z)
        
        near = wp.max(near, 0.0)
        
        t_near[tid] = near
        t_far[tid] = far
        
        if (near < far) and (far > 0.0):
            valid_mask[tid] = wp.int8(1)
        else:
            valid_mask[tid] = wp.int8(0)


def render_rays(model, origins, dirs, aabb_min, aabb_max, background_color, num_samples=256, perturb=True):
    N = origins.shape[0]
    device = origins.device
    
    use_warp_intersect = HAS_WARP and type(model.encoder).__name__ == "HashEncodingWarp"

    if use_warp_intersect:
        t_near = torch.empty(N, dtype=torch.float32, device=device)
        t_far = torch.empty(N, dtype=torch.float32, device=device)
        valid_mask_int = torch.empty(N, dtype=torch.int8, device=device)
        
        wp.launch(
            kernel=ray_aabb_intersection_kernel,
            dim=N,
            inputs=[
                wp.from_torch(origins), wp.from_torch(dirs),
                float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]),
                float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2]),
                wp.from_torch(t_near), wp.from_torch(t_far), wp.from_torch(valid_mask_int)
            ]
        )
        valid_mask = valid_mask_int.bool()
        t_near_valid = t_near[valid_mask]
        t_far_valid = t_far[valid_mask]
    else:
        inv_dir = 1.0 / (dirs + 1e-8)
        
        t0 = (aabb_min - origins) * inv_dir
        t1 = (aabb_max - origins) * inv_dir
        
        t_min = torch.minimum(t0, t1)
        t_max = torch.maximum(t0, t1)
        
        t_near = torch.max(t_min, dim=-1)[0]
        t_far = torch.min(t_max, dim=-1)[0]
        
        t_near = torch.clamp(t_near, min=0.0)
        valid_mask = (t_near < t_far) & (t_far > 0)
        t_near_valid = t_near[valid_mask]
        t_far_valid = t_far[valid_mask]
    
    rgb_map = background_color.unsqueeze(0).expand(N, 3).clone()
    
    if valid_mask.sum() == 0:
        return rgb_map
    
    o_valid = origins[valid_mask]
    d_valid = dirs[valid_mask]
    n_valid = o_valid.shape[0]
    
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    z_vals = t_near_valid.unsqueeze(-1) * (1.0 - t_vals) + t_far_valid.unsqueeze(-1) * t_vals
    
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    
    if perturb:
        t_rand = torch.rand_like(z_vals)
    else:
        t_rand = torch.ones_like(z_vals) * 0.5
        
    z_vals = lower + (upper - lower) * t_rand
    
    pts_world = o_valid.unsqueeze(1) + z_vals.unsqueeze(-1) * d_valid.unsqueeze(1)
    dirs_expand = d_valid.unsqueeze(1).expand(-1, num_samples, -1)
    
    pts_normalized = contract_to_unit_cube(
        pts_world.reshape(-1, 3), 
        aabb_min, 
        aabb_max
    )
    
    valid_pts_mask = (pts_normalized >= 0.0) & (pts_normalized <= 1.0)
    valid_pts_mask = torch.all(valid_pts_mask, dim=-1) 
    valid_pts_mask = valid_pts_mask.reshape(n_valid, num_samples)
    
    pts_query = torch.clamp(pts_normalized, 0.0, 1.0)
    
    dirs_flat = dirs_expand.reshape(-1, 3)
    
    rgb_flat, sigma_flat = model(pts_query.reshape(-1, 3), dirs_flat)
    rgb = rgb_flat.reshape(n_valid, num_samples, 3)
    sigma = sigma_flat.reshape(n_valid, num_samples)
    
    sigma = sigma * valid_pts_mask.float()
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    delta_last = (t_far_valid - z_vals[..., -1]).unsqueeze(-1)
    dists = torch.cat([dists, delta_last], dim=-1)
    
    alpha = 1.0 - torch.exp(-sigma * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((n_valid, 1), device=device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]
    
    weights = alpha * transmittance
    rgb_rendered = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
    acc_map = torch.sum(weights, dim=-1)
    
    rgb_final = rgb_rendered + background_color * (1.0 - acc_map.unsqueeze(-1))
    
    rgb_map[valid_mask] = rgb_final
    
    return rgb_map
