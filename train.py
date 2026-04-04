import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from config import Config
from data import NeRFDataset
from model import InstantNGP
from rendering import render_rays

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_SSIM = True
except Exception:
    compute_ssim = None
    HAS_SSIM = False

import imageio


def compute_scene_bounds(config):
    aabb_min = torch.tensor(config.AABB_MIN)
    aabb_max = torch.tensor(config.AABB_MAX)
    print(f"  Fixed Scene bounds: min={aabb_min.numpy()}, max={aabb_max.numpy()}")
    return aabb_min, aabb_max


def validate_and_save(model, dataset, aabb_min, aabb_max, config, step, best_psnr, run_dir):
    model.eval()
    val_indices = [0, 20, 40, 60, 80] 
    psnr_total = 0.0
    ssim_total = 0.0
    val_bg = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)
    
    with torch.no_grad():
        for idx in val_indices:
            if idx >= len(dataset): break
                
            rays_o, rays_d, target_rgba = dataset.get_rays_for_image(idx)
            H, W = dataset.H, dataset.W
            
            target_rgb = target_rgba[..., :3] * target_rgba[..., 3:4] + val_bg * (1.0 - target_rgba[..., 3:4])
            
            chunk_size = 4096
            rgb_pred_list = []
            
            flat_o = rays_o.reshape(-1, 3)
            flat_d = rays_d.reshape(-1, 3)
            
            for k in range(0, flat_o.shape[0], chunk_size):
                chunk_o = flat_o[k : k+chunk_size]
                chunk_d = flat_d[k : k+chunk_size]
                rgb_chunk = render_rays(
                    model, chunk_o, chunk_d, 
                    aabb_min.to(config.DEVICE), 
                    aabb_max.to(config.DEVICE),
                    background_color=val_bg,
                    num_samples=config.N_SAMPLES,
                    perturb=False 
                )
                rgb_pred_list.append(rgb_chunk)
            
            rgb_pred = torch.cat(rgb_pred_list, 0).reshape(H, W, 3)
            
            # 1. PSNR
            mse = F.mse_loss(rgb_pred, target_rgb)
            psnr = -10. * torch.log10(mse)
            psnr_total += psnr.item()
            
            # 2. SSIM
            if HAS_SSIM and compute_ssim is not None:
                img_pred_np = torch.clamp(rgb_pred, 0.0, 1.0).cpu().numpy()
                img_target_np = target_rgb.cpu().numpy()
                ssim_val = compute_ssim(img_target_np, img_pred_np, data_range=1.0, channel_axis=-1)
                ssim_total += ssim_val
            
    avg_psnr = psnr_total / len(val_indices)
    avg_ssim = (ssim_total / len(val_indices)) if HAS_SSIM and compute_ssim is not None else 0.0
    
    is_best = avg_psnr > best_psnr
    if is_best:
        best_psnr = avg_psnr
        ssim_str = f" | SSIM: {avg_ssim:.4f}" if HAS_SSIM and compute_ssim is not None else ""
        print(f"[Step {step}] New Best Validation PSNR: {best_psnr:.2f} dB{ssim_str}. Saving checkpoint.")
        torch.save(model.state_dict(), os.path.join(run_dir, "instant_ngp_best.pth"))
    else:
        ssim_str = f" | SSIM: {avg_ssim:.4f}" if HAS_SSIM and compute_ssim is not None else ""
        print(f"[Step {step}] Validation PSNR: {avg_psnr:.2f} dB{ssim_str}")
        
    return avg_psnr, avg_ssim, best_psnr


def evaluate(model, dataset, aabb_min, aabb_max, config, run_dir):
    model.eval()
    print(f"Starting Evaluation on {len(dataset)} images...")
    
    renders_dir = os.path.join(run_dir, "test_renders")
    os.makedirs(renders_dir, exist_ok=True)
    
    psnr_list = []
    ssim_list = []
    all_frames = [] 
    eval_bg = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)
    
    start_eval_time = time.time()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            rays_o, rays_d, target_rgba = dataset.get_rays_for_image(i)
            H, W = dataset.H, dataset.W
            
            target_rgb = target_rgba[..., :3] * target_rgba[..., 3:4] + eval_bg * (1.0 - target_rgba[..., 3:4])
            
            chunk_size = 4096
            rgb_pred_list = []
            
            flat_o = rays_o.reshape(-1, 3)
            flat_d = rays_d.reshape(-1, 3)
            
            for k in range(0, flat_o.shape[0], chunk_size):
                chunk_o = flat_o[k : k+chunk_size]
                chunk_d = flat_d[k : k+chunk_size]
                rgb_chunk = render_rays(
                    model, chunk_o, chunk_d, 
                    aabb_min.to(config.DEVICE), 
                    aabb_max.to(config.DEVICE),
                    background_color=eval_bg, 
                    num_samples=config.N_SAMPLES,
                    perturb=False 
                )
                rgb_pred_list.append(rgb_chunk)
            
            rgb_pred = torch.cat(rgb_pred_list, 0).reshape(H, W, 3)
            
            # 1. PSNR
            mse = F.mse_loss(rgb_pred, target_rgb)
            psnr = -10. * torch.log10(mse)
            psnr_list.append(psnr.item())
            
            # 2. SSIM
            if HAS_SSIM and compute_ssim is not None:
                img_pred_np = torch.clamp(rgb_pred, 0.0, 1.0).cpu().numpy()
                img_target_np = target_rgb.cpu().numpy()
                ssim_val = compute_ssim(img_target_np, img_pred_np, data_range=1.0, channel_axis=-1)
                ssim_list.append(ssim_val)
            
            img_np = (torch.clamp(rgb_pred, 0.0, 1.0).cpu().numpy() * 255).astype(__import__('numpy').uint8)
            all_frames.append(img_np)
            imageio.imwrite(os.path.join(renders_dir, f"test_{i:03d}.png"), img_np)
            
            if i % 10 == 0 or i == len(dataset) - 1:
                ssim_str = f" | SSIM = {ssim_list[-1]:.4f}" if HAS_SSIM and compute_ssim is not None else ""
                print(f"  Test Img {i}/{len(dataset)}: PSNR = {psnr.item():.2f} dB{ssim_str}")
                
    total_eval_time = time.time() - start_eval_time
    fps = len(dataset) / total_eval_time
    
    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list) if (HAS_SSIM and compute_ssim is not None and len(ssim_list)>0) else 0.0
    
    ssim_summary_str = f" | Mean SSIM: {mean_ssim:.4f}" if HAS_SSIM and compute_ssim is not None else ""
    print(f"Evaluation Complete. Mean PSNR: {mean_psnr:.2f} dB{ssim_summary_str} | Inference Speed: {fps:.2f} FPS")
    
    print("Saving GIF video...")
    imageio.mimsave(os.path.join(run_dir, "video.gif"), all_frames, fps=30)
    print(f"Video saved to '{run_dir}/video.gif'")
    
    return mean_psnr, mean_ssim, fps


def generate_graphs(metrics, run_dir):
    if not HAS_MATPLOTLIB or plt is None:
        return
    
    print("Generating performance graphs...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_times"], metrics["train_psnr"], alpha=0.3, label="Train PSNR (per batch)", color='blue')
    if metrics["val_psnr"]:
        plt.plot(metrics["val_times"], metrics["val_psnr"], marker='o', label="Validation PSNR", color='orange', linewidth=2)
    plt.title("Convergence: PSNR vs. Time")
    plt.xlabel("Cumulative Training Time (Seconds)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_psnr_vs_time.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_steps"], metrics["train_psnr"], alpha=0.3, label="Train PSNR (per batch)", color='blue')
    if metrics["val_psnr"]:
        plt.plot(metrics["val_steps"], metrics["val_psnr"], marker='o', label="Validation PSNR", color='orange', linewidth=2)
    plt.title("Convergence: PSNR vs. Step")
    plt.xlabel("Training Step")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_psnr_vs_step.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_steps"], metrics["step_times_ms"], alpha=0.6, color='green')
    
    avg_step_time = sum(metrics["step_times_ms"]) / len(metrics["step_times_ms"])
    plt.axhline(y=avg_step_time, color='r', linestyle='--', label=f'Avg: {avg_step_time:.2f} ms')
    
    plt.title("Training Throughput: Time per Step")
    plt.xlabel("Training Step")
    plt.ylabel("Time per Step (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, "graph_throughput.png"))
    plt.close()

    if HAS_SSIM and metrics["val_ssim"]:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["val_steps"], metrics["val_ssim"], marker='o', label="Validation SSIM", color='purple', linewidth=2)
        plt.title("Convergence: SSIM vs. Step")
        plt.xlabel("Training Step")
        plt.ylabel("SSIM")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, "graph_ssim_vs_step.png"))
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Instant NGP PyTorch/Warp Implementation")
    parser.add_argument('--exp_name', type=str, default="run_baseline", help="Name of the experiment run folder")
    parser.add_argument('--data_root', type=str, default=Config.DATA_ROOT, help="Path to dataset")
    parser.add_argument('--use_warp', action='store_true', help="Enable Warp backend")
    parser.add_argument('--iterations', type=int, default=Config.ITERATIONS, help="Number of training iterations")
    parser.add_argument('--val_interval', type=int, default=Config.VAL_INTERVAL, help="Validation interval")
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help="Batch size (rays per step)")
    parser.add_argument('--lr', type=float, default=Config.LR, help="Learning rate")
    parser.add_argument('--n_samples', type=int, default=Config.N_SAMPLES, help="Number of samples per ray")
    parser.add_argument('--l', type=int, default=Config.L, dest='L', help="Number of hash grid levels")
    parser.add_argument('--f', type=int, default=Config.F, dest='F', help="Feature dimension per level")
    parser.add_argument('--t', type=int, default=Config.T, dest='T', help="Hash table size")
    parser.add_argument('--aabb_min', type=float, nargs=3, default=Config.AABB_MIN, help="AABB min coords (e.g. -1.5 -1.5 -1.5)")
    parser.add_argument('--aabb_max', type=float, nargs=3, default=Config.AABB_MAX, help="AABB max coords (e.g. 1.5 1.5 1.5)")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    run_dir = os.path.join("runs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"=== Instant NGP: Optimized & Paper-Corrected ===")
    print(f"Saving outputs to: {run_dir}/")
    
    if HAS_SSIM:
        print("SSIM Tracking: ENABLED (scikit-image found)")
    else:
        print("SSIM Tracking: DISABLED (install scikit-image to enable)")
    
    config = Config()
    
    config.DATA_ROOT = args.data_root
    config.USE_WARP = args.use_warp
    config.ITERATIONS = args.iterations
    config.VAL_INTERVAL = args.val_interval
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.N_SAMPLES = args.n_samples
    config.L = args.L
    config.F = args.F
    config.T = args.T
    config.AABB_MIN = args.aabb_min
    config.AABB_MAX = args.aabb_max

    if not HAS_MATPLOTLIB:
        print("matplotlib not available; graphs disabled.")

    print(f"Backend: {'Warp' if config.USE_WARP else 'PyTorch'}")
    
    metrics = {
        "config": vars(args),
        "train_steps": [],
        "train_times": [],
        "train_psnr": [],
        "step_times_ms": [],
        "val_steps": [],
        "val_times": [],
        "val_psnr": [],
        "val_ssim": [],
        "eval_mean_psnr": 0.0,
        "eval_mean_ssim": 0.0,
        "eval_fps": 0.0
    }
    
    train_dataset = NeRFDataset(config.DATA_ROOT, split='train', device=config.DEVICE)
    val_dataset = NeRFDataset(config.DATA_ROOT, split='val', device=config.DEVICE)
    test_dataset = NeRFDataset(config.DATA_ROOT, split='test', device=config.DEVICE)
    
    aabb_min, aabb_max = compute_scene_bounds(config)
    train_dataset.shuffle()
    
    model = InstantNGP(config).to(config.DEVICE)
    if not config.USE_WARP:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=True)
    else:
        print("Skipping torch.compile (Warp Backend)...")
    
    params_decay = []
    params_no_decay = []
    
    for name, param in model.named_parameters():
        if "embeddings" in name:
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    optimizer = torch.optim.Adam(
        [
            {'params': params_no_decay, 'weight_decay': 0.0},
            {'params': params_decay, 'weight_decay': 1e-6}
        ],
        lr=config.LR, 
        betas=(config.ADAM_BETA1, config.ADAM_BETA2),
        eps=config.ADAM_EPS
    )
    
    def lr_lambda(step):
        if step < 60000:
            return 1.0
        elif step < 80000:
            return 0.33
        else:
            return 0.11

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    ray_idx = 0
    aabb_min_gpu = aabb_min.to(config.DEVICE)
    aabb_max_gpu = aabb_max.to(config.DEVICE)
    
    best_psnr = -1.0 
    
    print("Starting Training...")
    
    train_start_time = time.time()
    cumulative_train_time = 0.0
    
    for step in range(config.ITERATIONS):
        step_start_t = time.time()
        
        if ray_idx + config.BATCH_SIZE > len(train_dataset):
            train_dataset.shuffle()
            ray_idx = 0
            
        batch_o = train_dataset.rays_o[ray_idx : ray_idx + config.BATCH_SIZE]
        batch_d = train_dataset.rays_d[ray_idx : ray_idx + config.BATCH_SIZE]
        batch_rgba = train_dataset.target_rgba[ray_idx : ray_idx + config.BATCH_SIZE]
        ray_idx += config.BATCH_SIZE
        
        if config.RANDOM_BG_TRAIN:
            bg_color = torch.rand(3, device=config.DEVICE)
        else:
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=config.DEVICE)

        batch_rgb_pixels = batch_rgba[..., :3]
        batch_alpha = batch_rgba[..., 3:4]
        batch_target_rgb = batch_rgb_pixels * batch_alpha + bg_color * (1.0 - batch_alpha)
        
        rgb_pred = render_rays(
            model, batch_o, batch_d, 
            aabb_min_gpu, aabb_max_gpu,
            background_color=bg_color,
            num_samples=config.N_SAMPLES,
            perturb=True # Keep random jitter ON during training to learn the space
        )
        
        loss = F.mse_loss(rgb_pred, batch_target_rgb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        step_dt = time.time() - step_start_t
        cumulative_train_time += step_dt
        
        if step % 200 == 0 or step == config.ITERATIONS - 1:
            psnr = -10. * torch.log10(loss)
            metrics["train_steps"].append(step)
            metrics["train_times"].append(cumulative_train_time)
            metrics["train_psnr"].append(psnr.item())
            metrics["step_times_ms"].append(step_dt * 1000)
            
            print(f"[Step {step:05d}] Loss: {loss.item():.5f} | PSNR: {psnr.item():.2f}dB | Time: {step_dt*1000:.1f}ms/step | LR: {scheduler.get_last_lr()[0]:.2e}")
            
        if step > 0 and step % config.VAL_INTERVAL == 0:
            model.train(False) 
            current_psnr, current_ssim, best_psnr = validate_and_save(model, val_dataset, aabb_min_gpu, aabb_max_gpu, config, step, best_psnr, run_dir)
            
            metrics["val_steps"].append(step)
            metrics["val_times"].append(cumulative_train_time)
            metrics["val_psnr"].append(current_psnr)
            metrics["val_ssim"].append(current_ssim)
            
            model.train(True) 
            torch.cuda.empty_cache()
            
    print("Training Complete. Saving Last Model...")
    torch.save(model.state_dict(), os.path.join(run_dir, "instant_ngp_last.pth"))
    
    print("Reloading Best Model for Final Evaluation...")
    try:
        model.load_state_dict(torch.load(os.path.join(run_dir, "instant_ngp_best.pth"), map_location=config.DEVICE))
    except Exception as e:
        print(f"Warning: Could not load best model ({e}), using last model instead.")
        
    eval_psnr, eval_ssim, eval_fps = evaluate(model, test_dataset, aabb_min, aabb_max, config, run_dir)
    
    metrics["eval_mean_psnr"] = eval_psnr
    metrics["eval_mean_ssim"] = eval_ssim
    metrics["eval_fps"] = eval_fps
    
    metrics_path = os.path.join(run_dir, "metrics.json")
    def to_serializable(obj):
        # Convert torch tensors / numpy types to Python primitives for JSON
        try:
            import torch as _torch
            if isinstance(obj, _torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                return obj.tolist()
        except Exception:
            pass

        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        return obj

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, default=to_serializable)
    print(f"Saved run metrics textually to: {metrics_path}")
    
    generate_graphs(metrics, run_dir)


if __name__ == "__main__":
    main()
