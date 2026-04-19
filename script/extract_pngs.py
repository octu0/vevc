import sys
import numpy as np
from PIL import Image

def parse_y4m(file_path, target_frames):
    with open(file_path, 'rb') as f:
        header = f.readline().decode('ascii')
        w, h = 0, 0
        for token in header.split():
            if token.startswith('W'): w = int(token[1:])
            elif token.startswith('H'): h = int(token[1:])
        
        frame_idx = 0
        while True:
            frame_header = f.readline()
            if not frame_header: break
            if not frame_header.startswith(b'FRAME'):
                print(f"Format error at frame {frame_idx}")
                break
            
            y_data = f.read(w * h)
            cb_data = f.read((w // 2) * (h // 2))
            cr_data = f.read((w // 2) * (h // 2))
            
            if frame_idx in target_frames:
                y = np.frombuffer(y_data, dtype=np.uint8).reshape((h, w))
                img = Image.fromarray(y, mode='L')
                img.save(f"frame_{frame_idx:04d}.png")
                print(f"Saved frame_{frame_idx:04d}.png")
                
            frame_idx += 1
            if frame_idx > max(target_frames): break

if __name__ == "__main__":
    y4m = sys.argv[1]
    targets = [int(x) for x in sys.argv[2].split(",")]
    parse_y4m(y4m, targets)
