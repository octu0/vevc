#!/usr/bin/env python3
"""
元画像とデコード画像の差分を分析し、残像の原因を特定する。

使い方:
  1. エンコード: swift run -c release vevc-enc -o a.vevc docs/sample_frames/frame_0*.png
  2. デコード:   swift run -c release vevc-dec -i a.vevc -o /tmp/decoded/
  3. 比較:       python3 script/compare_frames.py docs/sample_frames/ /tmp/decoded/ [diff_dir] [gop_size]

引数:
  orig_dir   - 元画像ディレクトリ (e.g. docs/sample_frames/)
  dec_dir    - デコード画像ディレクトリ (e.g. /tmp/decoded/)
  diff_dir   - 差分画像出力先 (デフォルト: /tmp/diff/)
  gop_size   - GOP サイズ (デフォルト: 15)

--- ログの見方 ---

各行は1フレームの品質指標を示す:

  Frame  - フレーム番号（0始まり）
  Type   - I = Intra Frame（GOPの先頭、最も高品質）, P = Predicted Frame（差分符号化）
  MAE    - Mean Absolute Error: 全ピクセルの平均誤差（小さいほど良い）
           - I-Frame: 通常 1〜3
           - P-Frame: 3〜6 が許容範囲、8以上は問題あり
  MaxErr - 最大誤差: 255 の場合、一部ピクセルが完全に破綻している可能性あり
  PSNR   - Peak Signal-to-Noise Ratio [dB]: 高いほど良い
           - 40 dB 以上: 非常に高品質
           - 30〜40 dB: 一般的な圧縮品質
           - 30 dB 以下: 目に見える劣化
  SSIM   - Structural Similarity Index: 人間の知覚に近い品質指標（0〜1）
           - 0.95 以上: 人間の目にはほぼ劣化が見えない
           - 0.85〜0.95: わずかな劣化が見える
           - 0.85 以下: 明確な残像・ゴーストが発生
  Ghost  - Ghost Score: 残像（ゴースト）の検知指標（0〜100%）
           - 誤差ピクセルの中で「前フレームの方がデコード画像に近い」ピクセルの割合
           - 30% 以上: 残像が発生している可能性が高い
           - 50% 以上: 強い残像が発生
  >10, >20, >30 - 誤差がN以上のピクセルの割合

  Top error rows/cols - 誤差が最も集中している行/列の位置と平均誤差
           - I/P境界付近のフレームでのみ表示
           - 特定の行/列に集中 → ブロック境界やエッジ処理の問題
           - 広範囲に分散 → 動き推定や量子化の全体的な精度不足

--- 解析の手順 ---

1. まず I-Frame (Type=I) の品質を確認 → これがベースライン
2. P-Frame の SSIM が GOP 後半でどこまで低下するか確認
   - 0.94 以上を維持できていれば良好
   - 0.85 以下まで落ちるなら残差累積（error drift）が深刻
3. Ghost Score が 30% 以上のフレームを確認 → 残像発生箇所
4. MaxErr=255 が連続する区間を確認 → 破綻ピクセルの存在
5. >10 が急に 90% 以上に跳ね上がるフレームを特定
   → そのフレームで大きな変化（動き推定の失敗）が発生
6. diff 画像 (/tmp/diff/diff_NNNN.png) を確認
   → 誤差の空間分布でエッジ/ブロック境界/動き方向を特定
7. Summary の P-Frame MAE trend を確認
   → first→last で増加傾向 = 累積劣化あり
"""
import sys
import os
from PIL import Image
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("Warning: skimage not found. SSIM will not be calculated.")
    print("  Install: pip install scikit-image")

orig_dir = sys.argv[1]    # docs/sample_frames/
dec_dir = sys.argv[2]     # /tmp/decoded/
diff_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/diff/"
gop_size = int(sys.argv[4]) if len(sys.argv) > 4 else 15
os.makedirs(diff_dir, exist_ok=True)

orig_files = sorted([f for f in os.listdir(orig_dir) if f.endswith('.png')])
dec_files = sorted([f for f in os.listdir(dec_dir) if f.endswith('.png')])

print(f"Original frames: {len(orig_files)}")
print(f"Decoded frames: {len(dec_files)}")
print(f"GOP size: {gop_size}")
n = min(len(orig_files), len(dec_files))

# ヘッダ
cols = f"{'Frame':>8} {'Type':>6} {'MAE':>8} {'MaxErr':>8} {'PSNR':>8}"
if HAS_SSIM:
    cols += f" {'SSIM':>8}"
cols += f" {'Ghost':>8} {'>10':>8} {'>20':>8} {'>30':>8}"
print(f"\n{cols}")
print("-" * (len(cols) + 10))

# サマリー用の蓄積変数
all_mae = []
all_psnr = []
all_ssim = []
all_ghost = []
iframe_mae = []
pframe_mae = []
ghost_warnings = []     # (frame_idx, ghost_score, ssim_val)

prev_orig = None  # 前フレームの元画像（Ghost Score計算用）

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

    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    mse = np.mean(diff ** 2)
    psnr_val = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

    # SSIM
    ssim_val = None
    if HAS_SSIM:
        ssim_val = ssim(orig, dec, data_range=255.0, channel_axis=-1, win_size=3)

    # Ghost Score: 誤差ピクセルのうち、前フレームの方がデコード画像に近いピクセルの割合
    ghost_score = 0.0
    if prev_orig is not None:
        prev_h = min(prev_orig.shape[0], h)
        prev_w = min(prev_orig.shape[1], w)
        # 誤差が閾値以上のピクセルを対象
        pixel_err = np.max(abs_diff[:prev_h, :prev_w, :], axis=2)
        error_mask = pixel_err > 10
        error_count = np.sum(error_mask)
        if error_count > 0:
            # デコード画像と現フレームの距離
            dist_current = np.sum(np.abs(dec[:prev_h, :prev_w, :] - orig[:prev_h, :prev_w, :]), axis=2)
            # デコード画像と前フレームの距離
            dist_prev = np.sum(np.abs(dec[:prev_h, :prev_w, :] - prev_orig[:prev_h, :prev_w, :]), axis=2)
            # 前フレームの方が近い = ゴーストピクセル
            ghost_pixels = np.sum((dist_prev[error_mask] < dist_current[error_mask]))
            ghost_score = ghost_pixels / error_count * 100

    total_pixels = h * w
    gt10 = np.sum(np.max(abs_diff, axis=2) > 10) / total_pixels * 100
    gt20 = np.sum(np.max(abs_diff, axis=2) > 20) / total_pixels * 100
    gt30 = np.sum(np.max(abs_diff, axis=2) > 30) / total_pixels * 100

    frame_type = "I" if i % gop_size == 0 else "P"

    # 行出力
    line = f"{i:>8} {frame_type:>6} {mae:>8.2f} {max_err:>8.0f} {psnr_val:>8.2f}"
    if HAS_SSIM:
        line += f" {ssim_val:>8.4f}"
    line += f" {ghost_score:>7.1f}% {gt10:>7.1f}% {gt20:>7.1f}% {gt30:>7.1f}%"

    # 警告フラグ
    flags = []
    if ghost_score >= 30:
        flags.append("GHOST")
    if ssim_val is not None and ssim_val < 0.94:
        flags.append("LOW_SSIM")
    if flags:
        line += f"  ⚠ {','.join(flags)}"
        ghost_warnings.append((i, ghost_score, ssim_val, mae))

    print(line)

    # サマリー蓄積
    all_mae.append(mae)
    all_psnr.append(psnr_val)
    all_ghost.append(ghost_score)
    if ssim_val is not None:
        all_ssim.append(ssim_val)
    if frame_type == "I":
        iframe_mae.append(mae)
    else:
        pframe_mae.append(mae)

    # 差分画像を10x増幅して出力 (見やすくする)
    diff_vis = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(diff_dir, f"diff_{i:04d}.png"))

    # I-Frameとその前後のP-Frame、差分の空間分布を確認
    is_near_iframe = (i % gop_size == 0) or (i % gop_size == 1) or (i % gop_size == gop_size - 1)
    if i <= 3 or is_near_iframe:
        row_err = np.mean(np.max(abs_diff, axis=2), axis=1)
        col_err = np.mean(np.max(abs_diff, axis=2), axis=0)
        top_rows = np.argsort(row_err)[-5:][::-1]
        top_cols = np.argsort(col_err)[-5:][::-1]
        print(f"  Top error rows: {[(r, f'{row_err[r]:.1f}') for r in top_rows]}")
        print(f"  Top error cols: {[(c, f'{col_err[c]:.1f}') for c in top_cols]}")

    prev_orig = orig

# --- サマリー ---
print("\n--- Summary ---")
print(f"  Avg MAE  : {np.mean(all_mae):.2f}")
print(f"  Avg PSNR : {np.mean(all_psnr):.2f} dB")
if all_ssim:
    print(f"  Avg SSIM : {np.mean(all_ssim):.4f}")
    low_ssim_count = sum(1 for s in all_ssim if s < 0.94)
    print(f"  Low SSIM (<0.94) frames: {low_ssim_count}/{len(all_ssim)}")
if all_ghost:
    ghost_frames = sum(1 for g in all_ghost if g >= 30)
    print(f"  Ghost detected (>=30%) frames: {ghost_frames}/{len(all_ghost)}")
if iframe_mae:
    print(f"  I-Frame MAE (avg): {np.mean(iframe_mae):.2f}")
if pframe_mae:
    print(f"  P-Frame MAE (avg): {np.mean(pframe_mae):.2f}")
    print(f"  P-Frame MAE trend: first={pframe_mae[0]:.2f}, last={pframe_mae[-1]:.2f}, max={max(pframe_mae):.2f}")

# 問題フレームの詳細
if ghost_warnings:
    print(f"\n--- Problem Frames ({len(ghost_warnings)} detected) ---")
    for (fidx, gs, sv, m) in ghost_warnings:
        ssim_str = f"SSIM={sv:.4f}" if sv is not None else "SSIM=N/A"
        print(f"  Frame {fidx:>4}: Ghost={gs:.1f}% {ssim_str} MAE={m:.2f}")
print("---")

print(f"\nDone. Diff images saved to {diff_dir}")
