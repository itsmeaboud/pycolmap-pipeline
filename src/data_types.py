from enum import Enum
import numpy as np
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union, List
from jaxtyping import Float64, Float32, UInt8, Int, Float
from src.utils import qvec2rotmat

PathLike = Union[Path | str]

 
class CameraModelType(Enum):
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    PINHOLE = "PINHOLE"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    OPNECV = "OPENCV"
    
class MatcherType(Enum):
    EXHAUSTIVE = "EXHAUSTIVE"
    SEQUENTIAL = "SEQUENTIAL"
    SPATIAL = "SPATIAL"
    
class ExtractorType(Enum):
    SIFT = "SIFT"
    ALIKED = "ALIKED"
    
    
@dataclass
class SparseConfig:
    images_dir: Path
    output_dir: Path
    db_dir: Path
    camera_model: CameraModelType = CameraModelType.SIMPLE_RADIAL
    matcher_type: MatcherType = MatcherType.EXHAUSTIVE
    extractor_type: ExtractorType = ExtractorType.SIFT
    
    
@dataclass
class DenseConfig:
    images_dir: Path
    output_dir: Path
    mvs_dir: Path
    

@dataclass
class Intrinsic:
    K_mat : Float[np.ndarray, "3 3"]
    height : int
    width : int
    fx : float | None = None
    fy : float | None = None
    cx : float | None = None
    cy : float | None = None
    
    
    def __post_init__(self):
        
        self.fx = self.K_mat[0, 0]
        self.fy = self.K_mat[1, 1]
        self.cx = self.K_mat[0, 2]
        self.cy = self.K_mat[1, 2]

@dataclass
class Extrinsic :

    T_w2c : Float[np.ndarray, "4 4"]
    T_c2w : Float[np.ndarray, "4 4"]
    
    def transform(self, T_ref: Float[np.ndarray, "4 4"]):
        
        self.T_w2c = self.T_w2c @ np.linalg.inv(T_ref)
        self.T_c2w = np.linalg.inv(self.T_w2c)
        
    @classmethod
    def from_quat(
        cls,
        qvec: Float[np.ndarray, "4 1"],
        tvec: Float[np.ndarray, "3 1"],
        ):
        
        T_w2c = np.eye(4, dtype = np.float32)
        T_w2c[:3, :3] = qvec2rotmat(qvec)
        T_w2c[:3, 3] = tvec
        
        T_c2w = np.linalg.inv(T_w2c)

        
        return cls(
            T_w2c = T_w2c,
            T_c2w = T_c2w,       
        )
        
    
    @property
    def R_w2c(self) -> np.ndarray:
        return self.T_w2c[:3, :3]
    
    @property
    def t_w2c(self) -> np.ndarray:
        return self.T_w2c[:3, 3]
    
    @property
    def R_c2w(self) -> np.ndarray:
        return self.T_c2w[:3, :3]
    
    @property
    def t_c2w(self) -> np.ndarray:
        return self.T_c2w[:3, 3]
    
    
         
@dataclass
class ImageData:
    idx : int
    id : int
    name : str
    path : PathLike
    pts2d_inliners : Float32[np.ndarray, "m_inliners 2"]
    pts2d_inliners_id : Float32[np.ndarray, "m_inliners 2"]
    pts2d_outliers : Float32[np.ndarray, "n_outliers 2"]
    extrinsic : Extrinsic
    summary: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            idx = data['frame_idx'],
            id = data['image_id'],
            name = data['image_name'],
            path = data['image_path'],
            pts2d_inliners = data['pts2d_inliners'],
            pts2d_inliners_id = data['pts2d_inliners_id'],
            pts2d_outliers = data['pts2d_outliers'],
            extrinsic = data['extrinsic'],
            summary = data['summary']
        )

@dataclass
class CameraData:
    id : int
    model : str
    intrinsic : Intrinsic

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id = data['camera_id'],
            model = data['model'],
            intrinsic = data['intrinsic']
        )
    

@dataclass
class Frame:

    image : ImageData
    camera : CameraData
    
    @property
    def summary(self):
        '''
        name_text = f"Image name: {self.image.name}"
        id_text = f"Image ID: {self.image.id}"
        inliners_text = f"Triangulated: {len(self.image.pts2d_inliners)} / {len(self.image.pts2d_inliners) + len(self.image.pts2d_outliers)}"
        model_text = f"Camera model: {self.camera.model}"
        dim_text = f"Dimensions: ({self.camera.intrinsic.height},{self.camera.intrinsic.width})"
        fov_text = f"FOVs: ({self.camera.intrinsic.fx},{self.camera.intrinsic.fy})"
        principal_pts_text = f"Principal points: ({self.camera.intrinsic.cx},{self.camera.intrinsic.cx})"
        extrinsic_text = f"Extrinsic: {self.image.extrinsic.T_c2w}"
        intrinsic_tex = f"Intrinsic: {self.camera.intrinsic.K_mat}"
        '''


        T_c2w_str = np.array2string(self.image.extrinsic.T_c2w, precision=4, suppress_small=True)
        K_mat_str = np.array2string(self.camera.intrinsic.K_mat, precision=2, suppress_small=True)

        # 2. Push the text perfectly flush to the left margin!
        return f"""
### 📷 Frame Summary: {self.image.name}

* **Image ID:** {self.image.id}
* **Camera Model:** {self.camera.model}
* **Triangulated Features:** {len(self.image.pts2d_inliners)} / {len(self.image.pts2d_inliners) + len(self.image.pts2d_outliers)}
* **Dimensions:** {self.camera.intrinsic.width}w x {self.camera.intrinsic.height}h
* **Focal Lengths (fx, fy):** ({self.camera.intrinsic.fx:.2f}, {self.camera.intrinsic.fy:.2f})
* **Principal Point (cx, cy):** ({self.camera.intrinsic.cx:.2f}, {self.camera.intrinsic.cy:.2f})

#### Extrinsic Matrix (T_c2w)
```text
{T_c2w_str}
"""
        
@dataclass
class PointCloudData :
    
    xyz: Float32[np.ndarray, "n_points 3"]
    rgb: UInt8[np.ndarray, "n_points 3"]
    ids: Int[np.ndarray, "n_points 3"]
    errors: Float32[np.ndarray, "n_points"]
    
    def transform(self, T_ref: Float[np.ndarray, "4 4"]):
        hom_pts = np.column_stack((self.xyz, np.ones(self.num_points)))
        transformed_pts = (T_ref @ hom_pts.T).T
        self.xyz = transformed_pts[:, :3]
          
    @property
    def has_points(self) -> bool:
        return len(self.xyz) > 0
    
    @property
    def num_points(self) -> int:
        return len(self.xyz)
    
    
@dataclass
class FrameList:
    
    _sequence: List[Frame] = field(default_factory = list)
    
    def images(self):
        for frame in self._sequence:
            yield frame.image
            
    def cameras(self):
        for frame in self._sequence:
            yield frame.camera
            
    def add_frame(self, frame: Frame):
        if isinstance(frame, Frame):
            self._sequence.append(frame)
            
    def transform(self, T_ref):
        
        for frame in self._sequence:
            frame.image.extrinsic.transform(T_ref)
    
    def sort_by_name(self):
        self._sequence.sort(key = lambda frame : frame.image.image_name)

    def sort_by_id(self):
        self._sequence.sort(key = lambda frame : frame.image.image_id)
            
    def __iter__(self) -> Iterator[Frame]:
        return iter(self._sequence)
    
    def __len__(self):
        return len(self._sequence)
    
    def __getitem__(self, idx):
        return self._sequence[idx]

@dataclass
class SceneReconstruction:
    
    pcd : PointCloudData
    frames_list: FrameList
    summary : str
    
    def transform(self, T_ref):
        
        self.pcd.transform(T_ref)
        self.frames_list.transform(T_ref)
        
    
        
        
    
    
    

        
