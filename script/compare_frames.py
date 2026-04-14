#!/usr/bin/env python3
"""
元動画とデコード動画の差分を分析し、残像の原因を特定する。
Y4Mファイルを入力として受け取り、エンコード・デコード・画像比較までを自動化します。

使い方:
  python3 script/compare_frames.py <input.y4m> [options]

処理フロー:
  1. swift run -c release vevc-enc で .vevc にエンコード
  2. swift run -c release vevc-dec で .y4m にデコード
  3. ffmpeg で入力動画と出力動画をそれぞれ連番PNGに展開
  4. MAE, PSNR, SSIM, Ghost Score 等を算出し、フレームごとに比較
  5. Ghost Score が閾値 (デフォルト30%) 以上のフレームを検出した場合、
     指定された出力ディレクトリに該当フレームの元画像、デコード画像、差分画像を保存する
"""

import sys
import os
import argparse
import subprocess
import shutil
from PIL import Image
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("Warning: skimage not found. SSIM will not be calculated.")
    print("  Install: pip install scikit-image")


def run_command(cmd, desc):
    print(f"[{desc}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error during {desc}:\n{result.stderr}")
        sys.exit(result.returncode)
    return result

def main():
    parser = argparse.ArgumentParser(description="VEVC Encode/Decode auto compare tool")
    parser.add_argument("input_y4m", help="Path to input .y4m file")
    parser.add_argument("--out_dir", default=".tmp/compare_out", help="Directory to save problem frames (Ghost detected)")
    parser.add_argument("--gop_size", type=int, default=15, help="GOP size for labeling I/P frames")
    parser.add_argument("--ghost_threshold", type=float, default=30.0, help="Ghost Score threshold to dump images (%)")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary files (vevc, y4m, png sequence)")
    
    args = parser.parse_args()

    input_y4m = args.input_y4m
    out_dir = args.out_dir
    gop_size = args.gop_size
    ghost_thresh = args.ghost_threshold

    if not os.path.exists(input_y4m):
        print(f"File not found: {input_y4m}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    tmp_dir = os.path.join(out_dir, "_tmp_work")
    os.makedirs(tmp_dir, exist_ok=True)
    
    tmp_vevc = os.path.join(tmp_dir, "temp.vevc")
    tmp_y4m = os.path.join(tmp_dir, "temp.y4m")
    orig_frames_dir = os.path.join(tmp_dir, "orig_frames")
    dec_frames_dir = os.path.join(tmp_dir, "dec_frames")
    
    os.makedirs(orig_frames_dir, exist_ok=True)
    os.makedirs(dec_frames_dir, exist_ok=True)

    try:
        # 1. Encode
        run_command(["swift", "run", "-c", "release", "vevc-enc", "-i", input_y4m, "-o", tmp_vevc], "Encode")

        # 2. Decode
        run_command(["swift", "run", "-c", "release", "vevc-dec", "-i", tmp_vevc, "-o", tmp_y4m], "Decode")

        # 3. Extract frames with ffmpeg
        run_command(["ffmpeg", "-y", "-v", "error", "-i", input_y4m, "-vsync", "0", os.path.join(orig_frames_dir, "frame_%04d.png")], "Extract Orig")
        run_command(["ffmpeg", "-y", "-v", "error", "-i", tmp_y4m, "-vsync", "0", os.path.join(dec_frames_dir, "frame_%04d.png")], "Extract Dec")

        # 4. Compare
        orig_files = sorted([f for f in os.listdir(orig_frames_dir) if f.endswith('.png')])
        dec_files = sorted([f for f in os.listdir(dec_frames_dir) if f.endswith('.png')])

        if not orig_files or not dec_files:
            print("Failed to extract PNG frames.")
            sys.exit(1)

        print(f"\nOriginal frames: {len(orig_files)}")
        print(f"Decoded frames: {len(dec_files)}")
        print(f"GOP size: {gop_size}")
        n = min(len(orig_files), len(dec_files))

        cols = f"{'Frame':>8} {'Type':>6} {'MAE':>8} {'MaxErr':>8} {'PSNR':>8}"
        if HAS_SSIM:
            cols += f" {'SSIM':>8}"
        cols += f" {'Ghost':>8} {'>10':>8} {'>20':>8} {'>30':>8}"
        print(f"\n{cols}")
        print("-" * (len(cols) + 10))

        all_mae = []
        all_psnr = []
        all_ssim = []
        all_ghost = []
        iframe_mae = []
        pframe_mae = []
        ghost_warnings = []

        prev_orig = None
        iframe_orig = None

        for i in range(n):
            orig_path = os.path.join(orig_frames_dir, orig_files[i])
            dec_path = os.path.join(dec_frames_dir, dec_files[i])

            orig_img = Image.open(orig_path).convert('RGB')
            dec_img = Image.open(dec_path).convert('RGB')
            
            orig = np.array(orig_img, dtype=np.float32)
            dec = np.array(dec_img, dtype=np.float32)

            h = min(orig.shape[0], dec.shape[0])
            w = min(orig.shape[1], dec.shape[1])
            orig = orig[:h, :w, :]
            dec = dec[:h, :w, :]

            diff = orig - dec
            abs_diff = np.abs(diff)

            mae = np.mean(abs_diff)
            max_err = np.max(abs_diff)
            mse = np.mean(diff ** 2)
            psnr_val = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

            ssim_val = None
            if HAS_SSIM:
                ssim_val = ssim(orig, dec, data_range=255.0, channel_axis=-1, win_size=3)

            frame_type = "I" if i % gop_size == 0 else "P"
            if frame_type == "I":
                iframe_orig = orig

            ghost_score = 0.0
            meta_img = None
            if prev_orig is not None:
                prev_h = min(prev_orig.shape[0], h)
                prev_w = min(prev_orig.shape[1], w)
                pixel_err = np.max(abs_diff[:prev_h, :prev_w, :], axis=2)
                error_mask = pixel_err > 10
                error_count = np.sum(error_mask)
                if error_count > 0:
                    dist_current = np.sum(np.abs(dec[:prev_h, :prev_w, :] - orig[:prev_h, :prev_w, :]), axis=2)
                    dist_prev = np.sum(np.abs(dec[:prev_h, :prev_w, :] - prev_orig[:prev_h, :prev_w, :]), axis=2)
                    
                    if iframe_orig is not None:
                        dist_iframe = np.sum(np.abs(dec[:prev_h, :prev_w, :] - iframe_orig[:prev_h, :prev_w, :]), axis=2)
                    else:
                        dist_iframe = np.full_like(dist_current, float('inf'))
                    
                    iframe_ghost_mask = (dist_iframe < dist_current) & error_mask
                    pframe_ghost_mask = (dist_prev < dist_current) & error_mask & ~iframe_ghost_mask
                    
                    ghost_pixels = np.sum(iframe_ghost_mask | pframe_ghost_mask)
                    ghost_score = ghost_pixels / error_count * 100
                    
                    if ghost_score >= ghost_thresh:
                        meta_img = np.zeros_like(dec[:prev_h, :prev_w, :], dtype=np.uint8)
                        meta_img[iframe_ghost_mask] = [255, 0, 0] # Red (I-Frame Ghost)
                        meta_img[pframe_ghost_mask] = [0, 255, 0] # Green (P-Frame Trailing)

            total_pixels = h * w
            gt10 = np.sum(np.max(abs_diff, axis=2) > 10) / total_pixels * 100
            gt20 = np.sum(np.max(abs_diff, axis=2) > 20) / total_pixels * 100
            gt30 = np.sum(np.max(abs_diff, axis=2) > 30) / total_pixels * 100

            line = f"{i:>8} {frame_type:>6} {mae:>8.2f} {max_err:>8.0f} {psnr_val:>8.2f}"
            if HAS_SSIM:
                line += f" {ssim_val:>8.4f}"
            line += f" {ghost_score:>7.1f}% {gt10:>7.1f}% {gt20:>7.1f}% {gt30:>7.1f}%"

            flags = []
            if ghost_score >= ghost_thresh:
                flags.append("GHOST")
            if ssim_val is not None and ssim_val < 0.94:
                flags.append("LOW_SSIM")
                
            if flags:
                line += f"  ⚠ {','.join(flags)}"
                
            # GHOST判定が出たら画像をダンプ
            if ghost_score >= ghost_thresh:
                ghost_warnings.append((i, ghost_score, ssim_val, mae))
                orig_out = os.path.join(out_dir, f"ghost_{i:04d}_orig.png")
                dec_out = os.path.join(out_dir, f"ghost_{i:04d}_dec.png")
                diff_out = os.path.join(out_dir, f"ghost_{i:04d}_diff.png")
                meta_out = os.path.join(out_dir, f"ghost_{i:04d}_meta.png")
                
                orig_img.save(orig_out)
                dec_img.save(dec_out)
                # 差分画像を10倍増幅して出力
                diff_vis = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
                Image.fromarray(diff_vis).save(diff_out)
                if meta_img is not None:
                    Image.fromarray(meta_img).save(meta_out)

            print(line)

            all_mae.append(mae)
            all_psnr.append(psnr_val)
            all_ghost.append(ghost_score)
            if ssim_val is not None:
                all_ssim.append(ssim_val)
            if frame_type == "I":
                iframe_mae.append(mae)
            else:
                pframe_mae.append(mae)

            prev_orig = orig
            
            # Flush stdout for real-time progress viewing
            sys.stdout.flush()

        # --- サマリー ---
        print("\n--- Summary ---")
        print(f"  Avg MAE  : {np.mean(all_mae):.2f}")
        print(f"  Avg PSNR : {np.mean(all_psnr):.2f} dB")
        if all_ssim:
            print(f"  Avg SSIM : {np.mean(all_ssim):.4f}")
            low_ssim_count = sum(1 for s in all_ssim if s < 0.94)
            print(f"  Low SSIM (<0.94) frames: {low_ssim_count}/{len(all_ssim)}")
        if all_ghost:
            ghost_frames = sum(1 for g in all_ghost if g >= ghost_thresh)
            print(f"  Ghost detected (>={ghost_thresh}%) frames: {ghost_frames}/{len(all_ghost)}")
        if iframe_mae:
            print(f"  I-Frame MAE (avg): {np.mean(iframe_mae):.2f}")
        if pframe_mae:
            print(f"  P-Frame MAE (avg): {np.mean(pframe_mae):.2f}")
            print(f"  P-Frame MAE trend: first={pframe_mae[0]:.2f}, last={pframe_mae[-1]:.2f}, max={max(pframe_mae):.2f}")

        if ghost_warnings:
            print(f"\n--- Problem Frames ({len(ghost_warnings)} detected) ---")
            for (fidx, gs, sv, m) in ghost_warnings:
                ssim_str = f"SSIM={sv:.4f}" if sv is not None else "SSIM=N/A"
                print(f"  Frame {fidx:>4}: Ghost={gs:.1f}% {ssim_str} MAE={m:.2f}")
            print(f"-> Dumped images to {out_dir}/ ghost_*")

    finally:
        if not args.keep_tmp and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()
