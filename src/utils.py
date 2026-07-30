from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union, List
from jaxtyping import Float64, Float32, UInt8, Int, Float

PathLike = Union[Path | str]

import numpy as np

def qvec2rotmat(qvec):
    """
    Converts a quaternion in [x, y, z, w] order to a 3x3 rotation matrix.
    Assumes the quaternion is normalized (unit length).
    """
    # Unpack the quaternion in the NEW order
    x, y, z, w = qvec
    
    # Pre-compute squares and cross terms
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    # Build the 3x3 rotation matrix using the standard formula
    return np.array([
        [1 - 2*(y2 + z2), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(x2 + z2), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(x2 + y2)]
    ])


    
    
        

        

    

    
        
     
    
    
    