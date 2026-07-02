#!/usr/bin/env python3
"""
layer0 / layer0+1 再生時のクロマ飽和（真っ赤/真っ青）問題の診断スクリプト。

エンコード → maxLayer 0/1/2 でデコード → フレーム毎の U/V 統計を出力し、
- 飽和が「徐々に進行」するか（＝参照ドリフト: オープンループMCの蓄積誤差）
- 「特定フレームで突然発生」するか（＝ビットストリーム/パースのバグ）
- I-frame境界（keyint）でリセットされるか
を切り分ける。

使い方:
  python3 script/analyze_chroma_drift.py a.y4m [--keyint 30] [--frames 120] [--skip-encode out.vevc]

出力の見方:
  dU/dV = フル解像度(layer2)デコード結果の平均U/Vとの差分（縮小レイヤーをフル基準と比較）
  sat%  = U or V が 250以上 or 5以下 のピクセル比率
  ドリフトなら |dU|/|dV| がP-frameごとに単調増加し、I-frameで0付近へ戻る。
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def run(cmd, desc):
    print(f"[{desc}] {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)


def read_y4m(path):
    """Y4M を (frames, meta) で返す。frames は [(Y,U,V)] の np.uint8 配列リスト。"""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", "replace").strip()
        w = h = None
        for tok in header.split()[1:]:
            if tok.startswith("W"):
                w = int(tok[1:])
            elif tok.startswith("H"):
                h = int(tok[1:])
        if w is None or h is None:
            raise ValueError(f"bad y4m header: {header}")
        cw, ch = (w + 1) // 2, (h + 1) // 2
        ysize, csize = w * h, cw * ch
        frames = []
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"FRAME"):
                raise ValueError(f"expected FRAME, got {line[:20]!r}")
            buf = f.read(ysize + 2 * csize)
            if len(buf) < ysize + 2 * csize:
                break
            y = np.frombuffer(buf, np.uint8, ysize, 0).reshape(h, w)
            u = np.frombuffer(buf, np.uint8, csize, ysize).reshape(ch, cw)
            v = np.frombuffer(buf, np.uint8, csize, ysize + csize).reshape(ch, cw)
            frames.append((y, u, v))
        return frames, (w, h)


def stats(u, v):
    sat = (
        np.count_nonzero((u >= 250) | (u <= 5)) + np.count_nonzero((v >= 250) | (v <= 5))
    ) / (u.size + v.size) * 100.0
    return u.mean(), v.mean(), sat


def chroma_energy(u, v):
    """彩度の指標: U/V の128からの平均偏差。デコード側がこの値で元動画より小さければ退色。"""
    du = np.abs(u.astype(np.int16) - 128).mean()
    dv = np.abs(v.astype(np.int16) - 128).mean()
    return du + dv


def upscale_nearest(plane, factor, h, w):
    """nearest近傍でfactor倍に拡大し (h, w) にクロップ。"""
    up = np.repeat(np.repeat(plane, factor, axis=0), factor, axis=1)
    return up[:h, :w]


def y_mae(dec_y, src_y, factor):
    """縮小レイヤーYをnearest拡大して元動画Yと比較したMAE。
    絶対値は解像度差でかさ上げされるが、GOP内での増加トレンドがノイズ蓄積を表す。"""
    h, w = src_y.shape
    up = upscale_nearest(dec_y, factor, h, w)
    return np.abs(up.astype(np.int16) - src_y.astype(np.int16)).mean()


def chroma_hf(u, v):
    """クロマ高周波エネルギー（参照不要のノイズ指標）。
    本来クロマは滑らかなので、この値がGOP内で増加し続けるなら
    ブロック状のクロマノイズが蓄積している。4近傍ラプラシアンの平均絶対値。"""
    total = 0.0
    for p in (u, v):
        p16 = p.astype(np.int16)
        lap = np.abs(4 * p16[1:-1, 1:-1] - p16[:-2, 1:-1] - p16[2:, 1:-1]
                     - p16[1:-1, :-2] - p16[1:-1, 2:])
        total += lap.mean()
    return total / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_y4m")
    ap.add_argument("--keyint", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0, help="0=all")
    ap.add_argument("--skip-encode", metavar="VEVC", help="既存の .vevc を使う")
    ap.add_argument("--bitrate", default="2000000")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="vevc_chroma_")
    vevc_path = args.skip_encode or os.path.join(tmp, "t.vevc")

    if not args.skip_encode:
        run(["swift", "run", "-c", "release", "vevc-enc",
             "-i", args.input_y4m, "-o", vevc_path,
             "-b", args.bitrate, "-keyint", str(args.keyint)], "encode")

    decoded = {}
    for layer in (0, 1, 2):
        out = os.path.join(tmp, f"l{layer}.y4m")
        run(["swift", "run", "-c", "release", "vevc-dec",
             "-i", vevc_path, "-o", out, "-maxLayer", str(layer)], f"decode L{layer}")
        decoded[layer], dims = read_y4m(out)
        print(f"layer{layer}: {len(decoded[layer])} frames {dims}", file=sys.stderr)

    src_frames, src_dims = read_y4m(args.input_y4m)
    print(f"source: {len(src_frames)} frames {src_dims}", file=sys.stderr)

    n = min(len(decoded[0]), len(decoded[1]), len(decoded[2]), len(src_frames))
    if args.frames:
        n = min(n, args.frames)

    # srcdU/srcdV = 元動画とL2(フル解像度)のU/V平均差。GOP内で単調に悪化して
    # I-frameで戻るならフル解像度のクロマがGOP内で劣化している（全レイヤー共通問題）。
    # chromaR = L2の彩度 / 元動画の彩度（1.0未満が続くと退色＝洗い流されたような色）。
    # yMAE  = 縮小レイヤーYをnearest拡大して元動画Yと比較（構造劣化。絶対値より
    #         GOP内での増加トレンドが重要）
    # cHF   = クロマ高周波エネルギー（参照不要のブロックノイズ指標）
    factor = {0: 4, 1: 2}
    print(f"{'frm':>4} {'gop':>4} | "
          f"{'srcdU':>6} {'srcdV':>6} {'chromaR':>7} | "
          f"{'L1dU':>5} {'L1dV':>5} {'L1yMAE':>7} {'L1cHF':>6} | "
          f"{'L0dU':>5} {'L0dV':>5} {'L0yMAE':>7} {'L0cHF':>6}")

    # GOP内位置別の集計（トレンド可視化用）
    agg = {L: {"ymae": {}, "chf": {}} for L in (0, 1)}
    src_chf_by_pos = {}

    for i in range(n):
        _, u2, v2 = decoded[2][i]
        ys, us, vs = src_frames[i]
        mu2, mv2, _ = stats(u2, v2)
        mus, mvs_, _ = stats(us, vs)
        sdu, sdv = mu2 - mus, mv2 - mvs_
        cr = chroma_energy(u2, v2) / max(chroma_energy(us, vs), 1e-6)
        gpos = i % args.keyint
        src_chf_by_pos.setdefault(gpos, []).append(chroma_hf(us, vs))
        row = (f"{i:>4} {gpos:>4} | "
               f"{sdu:6.1f} {sdv:6.1f} {cr:7.3f} |")
        for layer in (1, 0):
            yd, u, v = decoded[layer][i]
            mu, mv, _ = stats(u, v)
            du, dv = mu - mu2, mv - mv2
            ym = y_mae(yd, ys, factor[layer])
            ch = chroma_hf(u, v)
            agg[layer]["ymae"].setdefault(gpos, []).append(ym)
            agg[layer]["chf"].setdefault(gpos, []).append(ch)
            row += f" {du:5.1f} {dv:5.1f} {ym:7.2f} {ch:6.2f} |"
        print(row)

    print()
    print("=== GOP内位置別の平均（全GOP集計。位置とともに増加するならノイズ蓄積） ===")
    print(f"{'gop':>4} | {'L0yMAE':>7} {'L0cHF':>6} | {'L1yMAE':>7} {'L1cHF':>6} | {'srcCHF':>6}")
    positions = sorted(agg[0]["ymae"].keys())
    first_l0 = None
    last_l0 = None
    for gpos in positions:
        m0 = np.mean(agg[0]["ymae"][gpos])
        c0 = np.mean(agg[0]["chf"][gpos])
        m1 = np.mean(agg[1]["ymae"][gpos])
        c1 = np.mean(agg[1]["chf"][gpos])
        sc = np.mean(src_chf_by_pos[gpos])
        if first_l0 is None:
            first_l0 = (m0, c0)
        last_l0 = (m0, c0)
        print(f"{gpos:>4} | {m0:7.2f} {c0:6.2f} | {m1:7.2f} {c1:6.2f} | {sc:6.2f}")

    print()
    if first_l0 and last_l0:
        ym_ratio = last_l0[0] / max(first_l0[0], 1e-6)
        ch_ratio = last_l0[1] / max(first_l0[1], 1e-6)
        print(f"L0: GOP先頭→末尾で yMAE x{ym_ratio:.2f}, cHF x{ch_ratio:.2f}")
        print("判定: x1.2 を超えて単調増加していればP-frame毎の構造ノイズ蓄積"
              "（オープンループの分散蓄積）。I-frame(位置0)に戻ると回復する。")
        print("srcCHF列は元動画自体の高周波量（シーン変化の基準値）。")


if __name__ == "__main__":
    main()
