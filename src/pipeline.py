from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from src.data_types import ExtractorType
from .sparse import run_sparse_colmap, build_reconstruction_output, run_load_reconstruction

from .mvs import run_mvs_colmap
from .vis_util import create_rrd, stream_data
from uuid import uuid4

if TYPE_CHECKING:
    from .data_types import SparseConfig, DenseConfig

def run_sparse_pipeline(cfg: SparseConfig):
    
    recon_dict: dict = run_sparse_colmap(
        images_path = cfg.images_dir,
        output_path = cfg.output_dir,
        db_path = cfg.db_dir,
        camera_model = cfg.camera_model,
        matcher_type = cfg.matcher_type,
        extractor_type = cfg.extractor_type
        )
    
    for recon_num in recon_dict.keys():
        
        reconstruction_output = build_reconstruction_output(
            reconstruction = recon_dict[recon_num],
            images_path = cfg.images_dir,
            )

    
    return reconstruction_output


def run_load_pipeline(model_path: Path, images_path: Path):
    
    return run_load_reconstruction(model_path, images_path)
    
    
    
def run_mvs_pipeline(cfg: DenseConfig):
    
    run_mvs_colmap(
        images_dir = cfg.images_dir,
        output_dir = cfg.output_dir,
        mvs_dir = cfg.mvs_dir,
    )
    
if __name__ == "__main__":
    pass