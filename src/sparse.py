from optparse import Option
import pycolmap
from pathlib import Path
from typing import Union, Optional
from jaxtyping import Float32, Float64
import numpy as np
from src.data_types import (
    SceneReconstruction,
    ImageData,
    CameraData,
    Frame,
    FrameList, 
    PointCloudData,
    Extrinsic,
    Intrinsic,
    CameraModelType,
    MatcherType,
    ExtractorType,
    )
from src.utils import qvec2rotmat
import logging
from src.vis_util import  create_rrd
from uuid import uuid4
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def build_reconstruction_output(
    reconstruction: pycolmap.Reconstruction,
    images_path: Path,
    ) -> dict:
    
    recon_summary = reconstruction.summary()
    print(f"Reconstruction: {reconstruction.summary()}")
    # get 3d points
    num_points3d = reconstruction.num_points3D()
    
    if num_points3d > 0 :
        
        pts3d_xyz = np.zeros((num_points3d, 3), dtype = np.float32)
        pts3d_rgb = np.zeros((num_points3d, 3), dtype = np.uint8)
        pts3d_ids = np.zeros(num_points3d, dtype = np.int64)
        pts3d_error = np.zeros((num_points3d, 1), dtype = np.float32)
        
        
        for idx, (point3d_id, point3d) in enumerate(reconstruction.points3D.items()):

            pts3d_xyz[idx] = point3d.xyz
            pts3d_rgb[idx] = point3d.color
            pts3d_error[idx] = point3d.error
            pts3d_ids[idx] = point3d_id
            
            logger.info("Number of reconstructed 3D points: %f", len(pts3d_xyz))
            
            
    else:
        pts3d_xyz = np.empty((0, 3), dtype = np.float32)
        pts3d_rgb = np.empty((0, 3), dtype = np.uint8)
        pts3d_ids = np.empty(0, dtype = np.int64)
        pts3d_error = np.empty((0, 1), dtype = np.float32)
        
    # get cameras parameters
    
    frames_list = FrameList()
    for idx, (image_id, image) in enumerate(reconstruction.images.items()):
        
        image_summary = image.summary()
        
        frame_idx : int = idx
        image_name : str = image.name
        camera_id : int = image.camera_id
        image_path : Path = images_path / Path(image_name)
        
        # triangulated 2D points
        pts_pairs = [(point2d.xy, point2d.point3D_id) for point2d in image.points2D if point2d.has_point3D()]
        
        if pts_pairs:
            pts2d_inliners, pts2d_inliners_id = zip(*pts_pairs)
            
            pts2d_inliners = np.array(pts2d_inliners)
            pts2d_inliners_id = np.array(pts2d_inliners_id)
            
        else:
            pts2d_inliners = np.empty((0, 2), dtype = np.float64)
            pts2d_inliners_id = np.empty(0, dtype = np.int64)
            
        pcd = PointCloudData(
            xyz = pts3d_xyz,
            rgb = pts3d_rgb,
            ids = pts3d_ids,
            errors = pts3d_error,
        )
            
        # remaining 2D points
        pts2d_outliers = [point2d.xy for point2d in image.points2D if not point2d.has_point3D()] or None
        if pts2d_outliers:
            pts2d_outliers = np.array(pts2d_outliers)
            
        # Extract extrinsic
        pose: pycolmap.Rigid3d = image.cam_from_world()
        extrinsic = Extrinsic.from_quat(qvec = pose.rotation.quat, tvec = pose.translation)
    
        # Extract camera parameters
        camera: pycolmap.Camera = reconstruction.cameras[camera_id]
        model: str = camera.model_name
        height : int = camera.height
        width : int = camera.width
        K: Float32[np.ndarray, "3 3"] = camera.calibration_matrix()
        intrinsic = Intrinsic(
            K_mat = K,
            height = height,
            width = width,
            )
        
        image_data_dict = {
            'frame_idx' : idx,
            'image_id' : image_id,
            'image_name' : image_name,
            'image_path' : image_path,
            'pts2d_inliners': pts2d_inliners,
            'pts2d_inliners_id': pts2d_inliners_id,
            'pts2d_outliers': pts2d_outliers,
            'extrinsic': extrinsic,
            'summary': image_summary
        }
        image_data = ImageData.from_dict(image_data_dict)
        
        camera_data_dict = {
            'camera_id' : camera_id,
            'model': model,
            'intrinsic' : intrinsic,
        }
        camera_data = CameraData.from_dict(camera_data_dict)
        
        frame = Frame(
            image=image_data,
            camera=camera_data,
            )
        
        frames_list.add_frame(frame)
    
    
    return SceneReconstruction(
        pcd = pcd,
        frames_list = frames_list,
        summary = recon_summary
    )
        
        
def exhastive_matching(db_path: Path, matching_opts: pycolmap.FeatureMatchingOptions):
    
    pair_opt = pycolmap.ExhaustivePairingOptions()
    pair_opt.block_size = 50
    
    pycolmap.match_exhaustive(
        database_path = db_path,
        pairing_options = pair_opt,
        matching_options = matching_opts,       
    )
    
def sequential_matching(db_path: Path, matching_opts: pycolmap.FeatureMatchingOptions):
    
    pair_opt = pycolmap.SequentialPairingOptions()
    pair_opt.overlap = 10
    
    pycolmap.match_sequential(
        database_path = db_path,
        pairing_options = pair_opt,
        matching_options = matching_opts,
    )
    
def spatial_matching(db_path: Path, matching_opts: pycolmap.FeatureMatchingOptions) :
    
    pair_opt = pycolmap.SpatialMatchingOptions()
    pair_opt.ignore_z = True
    
    pycolmap.match_spatial(
        database_path = db_path,
        pairing_options = pair_opt,
        matching_options = matching_opts,
    )
    
    
           
def run_sparse_colmap(
    images_path: Path,
    output_path: Path,
    db_path: Path,
    camera_model: CameraModelType,
    matcher_type: MatcherType,
    extractor_type: ExtractorType = ExtractorType.SIFT,
) -> dict[int, pycolmap.Reconstruction]:
    
    """
        This function will run the colamp feature extraction -> matching -> reconstruction
        if the db_file exist, delete and create a new
    """
    if not output_path.exists():
        output_path.mkdir()
 
    if db_path.exists():
        db_path.unlink()   
        
        
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model.value
    

        
    extraction_opts = pycolmap.FeatureExtractionOptions()
    if extractor_type == ExtractorType.ALIKED:
        extraction_opts.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        extraction_opts.max_image_size = 1024
    else:
        extraction_opts.type = pycolmap.FeatureExtractorType.SIFT
    
    
    
    # Feature extraction
    pycolmap.extract_features(
        database_path = db_path,
        image_path = images_path,
        reader_options = reader_opts,
        extraction_options = extraction_opts,
        
    )
    
    # Feature matching
    matcher_dispatch = {
        MatcherType.EXHAUSTIVE : exhastive_matching,
        MatcherType.SEQUENTIAL : sequential_matching,
        MatcherType.SPATIAL : spatial_matching,
    }
    
    matching_opts = pycolmap.FeatureMatchingOptions()
    if extractor_type == ExtractorType.ALIKED:
        
        matching_opts.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    else:
        matching_opts.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    
    matcher_dispatch[matcher_type](db_path, matching_opts)
    
    # reconstruct
    reconstruction: dict[int, pycolmap.Reconstruction] = pycolmap.incremental_mapping(
        database_path = db_path,
        image_path = images_path,
        output_path = output_path,
        )
           
    if not reconstruction:
        raise ValueError("Reconstruction failed")

    return reconstruction
 
 
 
 
def run_load_reconstruction(
    model_path: Path,
    images_path : Path | None = None
    ):
     
    reconstruction = pycolmap.Reconstruction(str(model_path))
    
    return build_reconstruction_output(reconstruction, images_path)
    


 
        

