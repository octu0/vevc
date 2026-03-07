#!/usr/bin/env python3
"""
元画像とデコード画像の差分を分析し、残像の原因を特定する。
- フレームごとのMSE/PSNR/MAE
- 差分画像を増幅して出力
- 差分が大きいピクセルの位置分布
"""
import sys
import os
from PIL import Image
import numpy as np

orig_dir = sys.argv[1]    # docs/sample_frames/
dec_dir = sys.argv[2]     # /tmp/decoded/
diff_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/diff/"
os.makedirs(diff_dir, exist_ok=True)

orig_files = sorted([f for f in os.listdir(orig_dir) if f.endswith('.png')])
dec_files = sorted([f for f in os.listdir(dec_dir) if f.endswith('.png')])

print(f"Original frames: {len(orig_files)}")
print(f"Decoded frames: {len(dec_files)}")
n = min(len(orig_files), len(dec_files))

print(f"\n{'Frame':>8} {'Type':>6} {'MAE':>8} {'MaxErr':>8} {'PSNR':>8} {'>10':>8} {'>20':>8} {'>30':>8}")
print("-" * 80)

for i in range(n):
    orig_path = os.path.join(orig_dir, orig_files[i])
    dec_path = os.path.join(dec_dir, dec_files[i])
    
    orig = np.array(Image.open(orig_path).convert('RGB'), dtype=np.float32)
    dec = np.array(Image.open(dec_path).convert('RGB'), dtype=np.float32)
    
    # サイズが違う場合はcrop
    h = min(orig.shape[0], dec.shape[0])
    w = min(orig.shape[1], dec.shape[1])
    orig = orig[:h, :w, :]
    dec = dec[:h, :w, :]
    
    diff = orig - dec
    abs_diff = np.abs(diff)
    
    from skimage.metrics import structural_similarity as ssim
    
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    
    # Calculate SSIM
    # win_size is the side-length of the sliding window used in comparison.
    # We use channel_axis=-1 since the image shape is (H, W, 3).
    # Provide data_range as 255 for uint8/0-255 float images.
    ssim_val = ssim(orig, dec, data_range=255.0, channel_axis=-1, win_size=3)
    
    total_pixels = h * w
    gt10 = np.sum(np.max(abs_diff, axis=2) > 10) / total_pixels * 100
    gt20 = np.sum(np.max(abs_diff, axis=2) > 20) / total_pixels * 100
    gt30 = np.sum(np.max(abs_diff, axis=2) > 30) / total_pixels * 100
    
    frame_type = "I" if i % 15 == 0 else "P"
    print(f"{i:>8} {frame_type:>6} {mae:>8.2f} {max_err:>8.0f} {psnr:>8.2f} {ssim_val:>8.4f} {gt10:>7.1f}% {gt20:>7.1f}% {gt30:>7.1f}%")
    
    # 差分画像を10x増幅して出力 (見やすくする)
    diff_vis = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(diff_dir, f"diff_{i:04d}.png"))
    
    # I-Frameと直後のP-Frame数枚だけ、差分の空間分布を確認
    if i <= 3 or i == 14 or i == 15 or i == 16:
        # 差分が大きいエリアのヒートマップ（行ごと/列ごとの平均誤差）
        row_err = np.mean(np.max(abs_diff, axis=2), axis=1)
        col_err = np.mean(np.max(abs_diff, axis=2), axis=0)
        
        # 上位5行/列
        top_rows = np.argsort(row_err)[-5:][::-1]
        top_cols = np.argsort(col_err)[-5:][::-1]
        print(f"  Top error rows: {[(r, f'{row_err[r]:.1f}') for r in top_rows]}")
        print(f"  Top error cols: {[(c, f'{col_err[c]:.1f}') for c in top_cols]}")

print("\nDone. Diff images saved to", diff_dir)
