
import pycolmap
from pathlib import Path

def run_mvs_colmap(
    images_dir: Path,
    output_dir: Path,
    mvs_dir: Path,
):       

    
    mvs_dir.mkdir(parents = True, exist_ok = True)
    
    pycolmap.undistort_images(mvs_dir, output_dir, images_dir)
        
    pycolmap.patch_match_stereo(mvs_dir)
 
    pycolmap.stereo_fusion(
        output_path = mvs_dir / "dense.ply",
        workspace_path = mvs_dir,
        output_type = "bin"
        )
