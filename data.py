import os
import time
import json
import numpy as np
import torch
import imageio

class NeRFDataset:
    def __init__(self, root_dir, split='train', device='cpu'):
        self.root_dir = root_dir
        self.split = split
        self.device = device
        
        meta_path = os.path.join(root_dir, f'transforms_{split}.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
            
        print(f"Loading {split} dataset from {meta_path}...")
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.camera_angle_x = self.meta.get('camera_angle_x', 0.6194058656692505)
        
        self.images = [] 
        self.poses = []
        
        frames = self.meta['frames']
        
        print(f"  Processing {len(frames)} images...")
        start_t = time.time()
        
        for frame in frames:
            fname = os.path.join(root_dir, frame['file_path'] + '.png')
            try:
                im_data = imageio.imread(fname)
            except Exception:
                fname = os.path.join(root_dir, frame['file_path'].strip('./') + '.png')
                im_data = imageio.imread(fname)

            im_data = im_data.astype(np.float32) / 255.0
            
            if im_data.shape[-1] == 3:
                im_data = np.concatenate([im_data, np.ones_like(im_data[..., :1])], axis=-1)
            
            self.images.append(torch.from_numpy(im_data))
            self.poses.append(torch.from_numpy(np.array(frame['transform_matrix'], dtype=np.float32)))
            
        self.H, self.W = self.images[0].shape[:2]
        self.focal = 0.5 * self.W / np.tan(0.5 * self.camera_angle_x)
        
        if split == 'train':
            print("  Generating rays for training...")
            self.rays_o, self.rays_d, self.target_rgba = self.generate_all_rays()
            self.rays_o = self.rays_o.to(device)
            self.rays_d = self.rays_d.to(device)
            self.target_rgba = self.target_rgba.to(device)
            
        print(f"  Loaded {split} set in {time.time()-start_t:.2f}s")

    def generate_all_rays(self):
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32), 
            torch.arange(self.H, dtype=torch.float32), 
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal, 
            -(j - self.H * 0.5) / self.focal, 
            -torch.ones_like(i)
        ], -1)
        
        rays_o_list = []
        rays_d_list = []
        rgba_list = []
        
        for idx in range(len(self.poses)):
            pose = self.poses[idx]
            ray_d = dirs @ pose[:3, :3].T
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_o = pose[:3, 3].expand_as(ray_d)
            
            rays_o_list.append(ray_o.reshape(-1, 3))
            rays_d_list.append(ray_d.reshape(-1, 3))
            rgba_list.append(self.images[idx].reshape(-1, 4))
            
        return torch.cat(rays_o_list, 0), torch.cat(rays_d_list, 0), torch.cat(rgba_list, 0)
        
    def get_rays_for_image(self, idx):
        pose = self.poses[idx].to(self.device)
        i, j = torch.meshgrid(
            torch.arange(self.W, dtype=torch.float32, device=self.device), 
            torch.arange(self.H, dtype=torch.float32, device=self.device), 
            indexing='xy'
        )
        dirs = torch.stack([
            (i - self.W * 0.5) / self.focal, 
            -(j - self.H * 0.5) / self.focal, 
            -torch.ones_like(i)
        ], -1)
        ray_d = dirs @ pose[:3, :3].T
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_o = pose[:3, 3].expand_as(ray_d)
        
        return ray_o, ray_d, self.images[idx].to(self.device)

    def shuffle(self):
        if self.split == 'train':
            perm = torch.randperm(self.rays_o.shape[0], device=self.device)
            self.rays_o = self.rays_o[perm]
            self.rays_d = self.rays_d[perm]
            self.target_rgba = self.target_rgba[perm]
            
    def __len__(self):
        return self.rays_o.shape[0] if self.split == 'train' else len(self.images)
