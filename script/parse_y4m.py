import numpy as np

def read_y4m(filename):
    with open(filename, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        parts = header.split(' ')
        w, h = 0, 0
        for p in parts:
            if p.startswith('W'): w = int(p[1:])
            if p.startswith('H'): h = int(p[1:])
        
        frames = []
        while True:
            frame_header = f.readline()
            if not frame_header or not frame_header.startswith(b'FRAME'):
                break
            
            y = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape((h, w))
            u = np.frombuffer(f.read((w // 2) * (h // 2)), dtype=np.uint8).reshape((h // 2, w // 2))
            v = np.frombuffer(f.read((w // 2) * (h // 2)), dtype=np.uint8).reshape((h // 2, w // 2))
            frames.append(y)
        return frames

f1 = read_y4m('/Users/octu0/Downloads/ToS-4k-1080.y4m')
print("Read {} frames".format(len(f1)))
