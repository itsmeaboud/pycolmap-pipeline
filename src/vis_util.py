from src.data_types import Frame, ImageData, CameraData, PointCloudData, FrameList, SceneReconstruction
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List
from uuid import uuid4
import logging

try:
       import rerun as rr
       import rerun.blueprint as rrb
       HAS_RERUN = True
except ImportError:
       HAS_RERUN = False

logger = logging.getLogger(__name__)

    
def create_rrd(
       scene: SceneReconstruction,
       output_path: Path,
       recording_id: str | None = None,
       
) -> Path :
       
       recording_id = recording_id or str(uuid4())
       rrd_dir = output_path / "rrd"
       if not rrd_dir.exists():
              rrd_dir.mkdir()
                
       recording = rr.RecordingStream(
              application_id = "COLMAP-Rerun 6",
              recording_id = recording_id,
       )
       
       try:
              
              rrd_file = rrd_dir / f"rrd_{recording_id}.rrd"
              logger.info("rrd file save: %s", rrd_file)
              recording.save(path = rrd_file)
              parent_path = Path("/world")
              
              recording.log(str("summary"), rr.TextDocument(scene.summary), static = True)
              
              recording.log(
                     "/",
                     rr.ViewCoordinates.RDF,
                     static = True,
              )
                         
              num_images = len(scene.frames_list)
              #T_ref = frames_list[0].image.extrinsic.T_w2c
              #pcd.transform(T_ref)
       
              recording.log(
                     str(parent_path / "pts3d"),
                     rr.Points3D(
                            scene.pcd.xyz,
                            colors = scene.pcd.rgb,
                            class_ids = scene.pcd.ids
                     ),
                     static = True,
                     )
              
              for idx, frame in enumerate(scene.frames_list):
              
                     image = frame.image
                     #image.extrinsic.transform(T_ref)
                     camera = frame.camera
                     image_name = image.name
                     image_path = image.path
                     summary = image.summary
                     
                     recording.set_time("id",  sequence = image.id)
                     
                     img = Image.open(image_path)
                     img = np.asarray(img) / 255.

                     camera_path = parent_path / "camera"
                     recording.log(
                            str(camera_path),
                            rr.Transform3D(
                                   translation = image.extrinsic.t_c2w,
                                   mat3x3 = image.extrinsic.R_c2w,
                            ),
                     )

                     image_path = camera_path / "image"
                     recording.log(
                            str(image_path),
                            rr.Pinhole(
                                   resolution=[camera.intrinsic.width, camera.intrinsic.height],
                                   image_from_camera = camera.intrinsic.K_mat
                            ),
                     )
                     


                     recording.log(
                            str(image_path),
                            rr.Image(img),
                     )

                     recording.log(
                            str(image_path / "kps"),
                            rr.Points2D(positions = image.pts2d_inliners, class_ids = image.pts2d_inliners_id, colors = [0, 255, 0]
                                   ),
                            )
   
       finally:
              recording.disconnect()
              

def stream_data(
       scene: SceneReconstruction
): 
       
       rr.init("Reconstruction", spawn = True)
 
       
       parent_path = Path("/world")
       rr.log("recon_summary", rr.TextDocument(scene.summary))

       rr.log(
              "/",
              rr.ViewCoordinates.RDF,
              static = True,
       )
       rr.log(
              str(parent_path / "pts3d"),
              rr.Points3D(
                     scene.pcd.xyz,
                     colors = scene.pcd.rgb,
                     class_ids = scene.pcd.ids
              ),
              static = True,
              )

       for idx, frame in enumerate(scene.frames_list):

              image = frame.image
              camera = frame.camera
              image_name = image.name
              image_path = image.path
              summary = image.summary

              rr.set_time("idx",  sequence = image.id)

              img = Image.open(image_path)
              img = np.asarray(img) / 255 
              camera_path = parent_path / "camera"
              rr.log(
                     str(camera_path),
                     rr.Transform3D(
                            translation = image.extrinsic.t_c2w,
                            mat3x3 = image.extrinsic.R_c2w,
                     ),
              )
                     
              image_path = camera_path / "image"
              rr.log(
                     str(image_path),
                     rr.Pinhole(
                            resolution=[camera.intrinsic.width, camera.intrinsic.height],
                            image_from_camera = camera.intrinsic.K_mat
                     ),
              )
              rr.log(
                     str(image_path / "summary"),
                     rr.TextDocument(text = frame.summary, media_type = rr.MediaType.MARKDOWN),
              )   

              rr.log(
                     str(image_path),
                     rr.Image(img),
              )

              rr.log(
                     str(image_path / "kps"),
                     rr.Points2D(positions = image.pts2d_inliners, class_ids = image.pts2d_inliners_id, colors = [0, 255, 0]
                            ),
                     )
              

