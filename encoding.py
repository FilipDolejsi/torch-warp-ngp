import math
import torch
import torch.nn as nn
try:
    import warp as wp
    HAS_WARP = True
except Exception:
    wp = None
    HAS_WARP = False

class HashEncodingPyTorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config.L
        self.F = config.F
        self.T = config.T
        self.primes = [config.PRIME_1, config.PRIME_2, config.PRIME_3]
        b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
        self.resolutions = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
        self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)

        self.register_buffer('offsets', torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=torch.long))

    def forward(self, x):
        encoded_features = []
        
        for i, res in enumerate(self.resolutions):
            x_scaled = x * res
            x_floor = x_scaled.long()
            weights = x_scaled - x_floor.float()
            fx, fy, fz = weights[:, 0], weights[:, 1], weights[:, 2]
            
            c = x_floor.unsqueeze(1) + self.offsets.unsqueeze(0)
            
            h = ((c[..., 0] * self.primes[0]) ^ 
                 (c[..., 1] * self.primes[1]) ^ 
                 (c[..., 2] * self.primes[2])) % self.T
                 
            corners = self.embeddings[i][h]
            
            c00 = corners[:, 0] * (1-fx).unsqueeze(-1) + corners[:, 4] * fx.unsqueeze(-1)
            c01 = corners[:, 1] * (1-fx).unsqueeze(-1) + corners[:, 5] * fx.unsqueeze(-1)
            c10 = corners[:, 2] * (1-fx).unsqueeze(-1) + corners[:, 6] * fx.unsqueeze(-1)
            c11 = corners[:, 3] * (1-fx).unsqueeze(-1) + corners[:, 7] * fx.unsqueeze(-1)
            c0 = c00 * (1-fy).unsqueeze(-1) + c10 * fy.unsqueeze(-1)
            c1 = c01 * (1-fy).unsqueeze(-1) + c11 * fy.unsqueeze(-1)
            c = c0 * (1-fz).unsqueeze(-1) + c1 * fz.unsqueeze(-1)
            
            encoded_features.append(c)
            
        return torch.cat(encoded_features, dim=-1)


if HAS_WARP:
    wp.init()

    @wp.kernel
    def hash_grid_forward_kernel(
        inputs: wp.array(dtype=wp.float32, ndim=2),
        embeddings: wp.array(dtype=wp.float32, ndim=3),
        resolutions: wp.array(dtype=wp.int32),
        primes: wp.array(dtype=wp.int32),
        L: int, T: int, F: int,
        output: wp.array(dtype=wp.float32, ndim=2)
    ):
        tid = wp.tid()
        x_in, y_in, z_in = inputs[tid, 0], inputs[tid, 1], inputs[tid, 2]
        
        for l in range(L):
            res = float(resolutions[l])
            x_s, y_s, z_s = x_in * res, y_in * res, z_in * res
            x0, y0, z0 = int(wp.floor(x_s)), int(wp.floor(y_s)), int(wp.floor(z_s))
            wx, wy, wz = x_s - float(x0), y_s - float(y0), z_s - float(z0)
            
            feat_val = wp.vec2(0.0, 0.0)
            
            for i in range(8):
                dx = (i >> 2) & 1
                dy = (i >> 1) & 1
                dz = i & 1
                
                cx = x0 + dx
                cy = y0 + dy
                cz = z0 + dz
                
                h_idx = ((cx * primes[0]) ^ (cy * primes[1]) ^ (cz * primes[2])) % T
                
                weight = (1.0 - wx if dx == 0 else wx) * \
                         (1.0 - wy if dy == 0 else wy) * \
                         (1.0 - wz if dz == 0 else wz)
                          
                feat_val[0] += embeddings[l, h_idx, 0] * weight
                feat_val[1] += embeddings[l, h_idx, 1] * weight
            
            output[tid, l * F + 0] = feat_val[0]
            output[tid, l * F + 1] = feat_val[1]

    @wp.kernel
    def hash_grid_backward_kernel(
        grad_output: wp.array(dtype=wp.float32, ndim=2),
        inputs: wp.array(dtype=wp.float32, ndim=2),
        resolutions: wp.array(dtype=wp.int32),
        primes: wp.array(dtype=wp.int32),
        L: int, T: int, F: int,
        grad_embeddings: wp.array(dtype=wp.float32, ndim=3)
    ):
        tid = wp.tid()
        x_in, y_in, z_in = inputs[tid, 0], inputs[tid, 1], inputs[tid, 2]
        
        for l in range(L):
            res = float(resolutions[l])
            x_s, y_s, z_s = x_in * res, y_in * res, z_in * res
            x0, y0, z0 = int(wp.floor(x_s)), int(wp.floor(y_s)), int(wp.floor(z_s))
            wx, wy, wz = x_s - float(x0), y_s - float(y0), z_s - float(z0)
            
            g0 = grad_output[tid, l*F + 0]
            g1 = grad_output[tid, l*F + 1]
            
            for i in range(8):
                dx = (i >> 2) & 1
                dy = (i >> 1) & 1
                dz = i & 1
                
                cx = x0 + dx
                cy = y0 + dy
                cz = z0 + dz
                
                h_idx = ((cx * primes[0]) ^ (cy * primes[1]) ^ (cz * primes[2])) % T
                
                weight = (1.0 - wx if dx == 0 else wx) * \
                         (1.0 - wy if dy == 0 else wy) * \
                         (1.0 - wz if dz == 0 else wz)
                          
                wp.atomic_add(grad_embeddings, l, h_idx, 0, g0 * weight)
                wp.atomic_add(grad_embeddings, l, h_idx, 1, g1 * weight)

    class WarpHashFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, embeddings, resolutions, primes, config):
            ctx.config = config
            ctx.save_for_backward(inputs, embeddings, resolutions, primes)
            N = inputs.shape[0]
            L, T, F = embeddings.shape
            encoded = torch.empty((N, L * F), device=inputs.device, dtype=torch.float32)
            
            wp.launch(kernel=hash_grid_forward_kernel, dim=N, inputs=[
                wp.from_torch(inputs), wp.from_torch(embeddings), wp.from_torch(resolutions),
                wp.from_torch(primes), L, T, F, wp.from_torch(encoded)
            ])
            return encoded

        @staticmethod
        def backward(ctx, grad_output):
            inputs, embeddings, resolutions, primes = ctx.saved_tensors
            L, T, F = embeddings.shape
            N = inputs.shape[0]
            grad_embeddings = torch.zeros_like(embeddings)
            
            wp.launch(kernel=hash_grid_backward_kernel, dim=N, inputs=[
                wp.from_torch(grad_output.contiguous()), wp.from_torch(inputs), wp.from_torch(resolutions),
                wp.from_torch(primes), L, T, F, wp.from_torch(grad_embeddings)
            ])
            return None, grad_embeddings, None, None, None

    class HashEncodingWarp(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.L, self.T, self.F = config.L, config.T, config.F
            b = math.exp((math.log(config.N_MAX) - math.log(config.N_MIN)) / (config.L - 1))
            resolutions_list = [math.floor(config.N_MIN * (b ** i)) for i in range(config.L)]
            self.register_buffer('resolutions', torch.tensor(resolutions_list, dtype=torch.int32))
            self.register_buffer('primes', torch.tensor([config.PRIME_1, config.PRIME_2, config.PRIME_3]))
            self.embeddings = nn.Parameter(torch.zeros(self.L, self.T, self.F))
            nn.init.uniform_(self.embeddings, -1e-4, 1e-4)
            
        def forward(self, x):
            return WarpHashFunc.apply(x, self.embeddings, self.resolutions, self.primes, self.config)
