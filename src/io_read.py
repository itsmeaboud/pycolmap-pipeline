import numpy as np
from pathlib import Path
import rerun as rr

def read_dense_bin(file_path: Path):
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} doesn't exist")
    
    with open(file_path, "rb") as f:
        
        header_bytes = b""
        amper_found = 0
        
        # 1. READ THE HEADER (Stay in this loop until 3 ampersands are found)
        while amper_found < 3:
            byte = f.read(1)
            
            if not byte:
                raise EOFError("Unexpected end of file in the header")
            
            header_bytes += byte
            if byte == b"&":
                amper_found += 1
                
        header_str = header_bytes.decode('ascii').strip('&')
        print(f"Header found: {header_str}")
        
        width, height, channels = [int(val) for val in header_str.split('&')]
        
        # 3. READ THE DATA
        raw_data = np.fromfile(f, dtype=np.float32)
        arr = raw_data.reshape((height, width, channels))
        
        # 4. CLEAN UP SHAPE
        if channels == 1:
            # Correct squeeze syntax: either np.squeeze(arr, axis=-1) OR arr.squeeze(axis=-1)
            arr = arr.squeeze(axis=-1)
            
        return arr

if __name__ == "__main__":
    file_path = Path("//home/aboud/Projects/colamp_project/colmap_project/mvs_dir/stereo/depth_maps/00.png.geometric.bin")
    
    arr = read_dense_bin(file_path)
    
    rr.init("test")
    rr.log("image",
           rr.DepthImage(arr))
    rr.connect_grpc()
    print(f"Successfully loaded array with shape: {arr.shape}")
    print(arr)