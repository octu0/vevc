#!/usr/bin/env python3
import sys
import re

def check_recon(log_file_path):
    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    enc_lines = [l.strip() for l in lines if "[RECON_ENC]" in l]
    dec_lines = [l.strip() for l in lines if "[RECON_DEC]" in l]

    # Parse all ENC frames
    enc_map = {}
    for l in enc_lines:
        m = re.search(r"\[RECON_ENC\] frame=(\d+) y=([0-9a-f]+) cb=([0-9a-f]+) cr=([0-9a-f]+)", l)
        if m:
            f_idx = int(m.group(1))
            enc_map[f_idx] = (m.group(2), m.group(3), m.group(4))

    # Parse all DEC frames
    dec_map = {}
    for l in dec_lines:
        m = re.search(r"\[RECON_DEC\] frame=(\d+) y=([0-9a-f]+) cb=([0-9a-f]+) cr=([0-9a-f]+)", l)
        if m:
            f_idx = int(m.group(1))
            # Store the first occurring match for each frame in main run
            if f_idx not in dec_map:
                dec_map[f_idx] = (m.group(2), m.group(3), m.group(4))

    frame_indices = sorted(enc_map.keys())
    if not frame_indices:
        print("ERROR: No ENC recon entries found in log.")
        return 1

    print(f"Total encoded frames logged: {len(frame_indices)}")
    
    mismatches = []
    for f_idx in frame_indices:
        if f_idx not in dec_map:
            mismatches.append((f_idx, "MISSING_IN_DEC", enc_map[f_idx], None))
            continue
        
        e_y, e_cb, e_cr = enc_map[f_idx]
        d_y, d_cb, d_cr = dec_map[f_idx]
        
        if (e_y, e_cb, e_cr) != (d_y, d_cb, d_cr):
            mismatches.append((f_idx, "HASH_MISMATCH", enc_map[f_idx], dec_map[f_idx]))

    if mismatches:
        first = mismatches[0]
        f_idx, reason, e_hash, d_hash = first
        if reason == "MISSING_IN_DEC":
            print(f"FIRST MISMATCH at Frame {f_idx}: Missing from Decoder recon output!")
        else:
            e_y, e_cb, e_cr = e_hash
            d_y, d_cb, d_cr = d_hash
            diff_planes = []
            if e_y != d_y: diff_planes.append("Y")
            if e_cb != d_cb: diff_planes.append("Cb")
            if e_cr != d_cr: diff_planes.append("Cr")
            plane_str = "/".join(diff_planes)
            print(f"FIRST MISMATCH at Frame {f_idx} (Plane: {plane_str}):")
            print(f"  ENC -> Y:{e_y} Cb:{e_cb} Cr:{e_cr}")
            print(f"  DEC -> Y:{d_y} Cb:{d_cb} Cr:{d_cr}")
        print(f"Total mismatched frames: {len(mismatches)} / {len(frame_indices)}")
        return 1
    else:
        print(">>> ALL FRAMES & PLANES MATCH BIT-EXACTLY (ENC vs DEC) <<<")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_recon.py <log_file>")
        sys.exit(1)
    sys.exit(check_recon(sys.argv[1]))
