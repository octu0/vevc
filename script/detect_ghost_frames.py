import os
import sys
import numpy as np
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

def read_y4m(filename):
    frames = []
    with open(filename, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        parts = header.split(' ')
        w, h = 0, 0
        for p in parts:
            if p.startswith('W'): w = int(p[1:])
            if p.startswith('H'): h = int(p[1:])
        
        while True:
            frame_header = f.readline()
            if not frame_header or not frame_header.startswith(b'FRAME'):
                break
            
            y = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape((h, w))
            _ = f.read((w // 2) * (h // 2))
            _ = f.read((w // 2) * (h // 2))
            frames.append(y)
    return frames, w, h

def main():
    if len(sys.argv) < 3:
        print("Usage: detect_ghost_frames.py orig.y4m dec.y4m")
        sys.exit(1)
        
    orig_y4m = sys.argv[1]
    dec_y4m = sys.argv[2]
    
    print("Reading original...")
    orig_frames, w, h = read_y4m(orig_y4m)
    print("Reading decoded...")
    dec_frames, _, _ = read_y4m(dec_y4m)
    
    n = min(len(orig_frames), len(dec_frames))
    print(f"Comparing {n} frames...")
    
    ghost_thresh = 30.0
    gop_size = 15
    iframe_orig = None
    prev_orig = None
    
    for i in range(n):
        orig = orig_frames[i].astype(np.float32)
        dec = dec_frames[i].astype(np.float32)
        
        diff = orig - dec
        abs_diff = np.abs(diff)
        mae = np.mean(abs_diff)
        
        frame_type = "I" if i % gop_size == 0 else "P"
        if frame_type == "I":
            iframe_orig = orig

        ghost_score = 0.0
        if prev_orig is not None:
            pixel_err = abs_diff
            error_mask = pixel_err > 10
            error_count = np.sum(error_mask)
            if error_count > 0:
                dist_current = np.abs(dec - orig)
                dist_prev = np.abs(dec - prev_orig)
                
                if iframe_orig is not None:
                    dist_iframe = np.abs(dec - iframe_orig)
                else:
                    dist_iframe = np.full_like(dist_current, float('inf'))
                
                iframe_ghost_mask = (dist_iframe < dist_current) & error_mask
                pframe_ghost_mask = (dist_prev < dist_current) & error_mask & ~iframe_ghost_mask
                
                ghost_pixels = np.sum(iframe_ghost_mask | pframe_ghost_mask)
                ghost_score = ghost_pixels / error_count * 100
        
        prev_orig = orig
        
        if ghost_score > ghost_thresh or mae > 3.0:
            print(f"Frame {i:4d} [{frame_type}]: MAE={mae:5.2f} Ghost={ghost_score:5.1f}%")

if __name__ == "__main__":
    main()
