# COLMAP reconstruction pipeline

This project wraps a small, practical COLMAP workflow in Python. Give it a
folder of overlapping images and it can build a sparse reconstruction, turn a
sparse model into a dense point cloud, and open an existing model in Rerun for
inspection.

It is intended as a pipeline rather than a replacement for COLMAP itself:

```text
images → features → matching → sparse model → dense model
                              ↘ Rerun visualisation
```

## Getting started

The project uses [Pixi](https://pixi.prefix.dev/) to create the Python and CUDA
environment. Install Pixi first, then run commands from the repository root.
Pixi will create the environment automatically the first time a command is
run.

```bash
pixi install
```

The environment targets Linux with CUDA 12. A CUDA-capable NVIDIA setup is
therefore expected for the `pycolmap-cuda12` package and GPU reconstruction.

To see the available commands at any time:

```bash
pixi run sparse --help
pixi run mvs --help
pixi run load --help
```

## 1. Create a sparse reconstruction

Put the input photos in one directory. The photos should have good overlap;
consecutive frames from a video are a typical input. Then run:

```bash
pixi run sparse ./images ./output
```

This step extracts features, matches them, and runs incremental mapping. It
creates a COLMAP database at `./output/database.db` and writes one or more
sparse models below `./output/` (for example, `./output/0/`).

Useful options:

```bash
pixi run sparse <PATH_TO_IMAGES> <PATH_TO_OUTPUT> \
  --database <DATABASE_NAME.db> \
  --camera-model <CAMERA_MODEL> \
  --extractor <EXTRACTOR_TYPE> \
  --matcher <MATCHER_TYPE> \
  --visualize
```

The supported matchers are `EXHAUSTIVE`, `SEQUENTIAL`, and `SPATIAL`.
The supported extractors are `SIFT` and `ALIKED`. `SIMPLE_RADIAL`, `PINHOLE`,
and `SIMPLE_PINHOLE` are among the camera models accepted by the CLI. The
defaults are `SIMPLE_RADIAL`, `SIFT`, and `EXHAUSTIVE`.

Running the sparse command again with the same database filename replaces that
database, so use a different output directory or `--database` value if the
previous run needs to be kept.

## 2. Build a dense reconstruction

Choose the sparse model you want to densify. For the first reconstruction this
is commonly `./output/0`, not merely `./output`.

```bash
pixi run mvs -- ./images ./output/0 ./mvs
```

## 3. Inspect an existing sparse model

To load a model that already exists and stream it to the Rerun viewer:

```bash
pixi run load ./images ./output/0
```

## What the code does

`main.py` is the command-line entry point. It turns each command into a small
configuration object and passes it to `src/pipeline.py`.

- `src/sparse.py` contains feature extraction, matching, incremental mapping,
  and conversion of COLMAP's reconstruction into the project's data objects.
- `src/mvs.py` runs image undistortion, patch-match stereo, and stereo fusion.
- `src/data_types.py` defines the reconstruction, camera, frame, and point
  cloud data passed through the pipeline.
- `src/vis_util.py` writes or streams the reconstruction to Rerun.

The command layer stays deliberately thin, so the same pipeline functions can
also be imported and called from another Python script.

## Using the pipeline from Python

If you are working from a clone of this repository, you can use the pipeline
directly from a script in the repository root. Run the script through Pixi so
it uses the same environment as the command-line tools:

Here is the smallest typed example:

```bash

# Add pip to the Pixi sandbox and install the package in editable mode
pixi add pip
pixi run pip install -e .
```


```python
from pathlib import Path

from src.data_types import (
    CameraModelType,
    ExtractorType,
    MatcherType,
    SparseConfig,
)
from src.pipeline import run_sparse_pipeline

images = Path("./images").resolve()
output = Path("./output").resolve()

config = SparseConfig(
    images_dir=images,
    output_dir=output,
    db_path=output / "database.db",
    camera_model=CameraModelType.SIMPLE_RADIAL,
    matcher_type=MatcherType.SEQUENTIAL,
    extractor_type=ExtractorType.SIFT,
)

scene = run_sparse_pipeline(config)
print(scene.summary)
print(f"Reconstructed {scene.pcd.num_points} points")
```

### What's in the SceneReconstruction?

The pipeline abstracts away COLMAP's raw `.bin` files into explicitly typed Python dataclasses. The resulting `scene` object gives you direct access to the 3D points and camera data, so a downstream application can use camera poses, triangulated features, point positions, colours, and errors without parsing COLMAP's binary files itself.

- `scene.pcd`: A `PointCloudData` object containing the 3D geometry.
  - `scene.pcd.xyz`: The (N, 3) spatial coordinates.
  - `scene.pcd.rgb`: The (N, 3) RGB colors for each point.
  - `scene.pcd.errors`: The (N,) reprojection errors.
- `scene.frames`: A list of `FrameData` objects representing the solved cameras.
  - `scene.frames[0].camera`: The camera intrinsics (focal length, principal point).
  - `scene.frames[0].image.extrinsic`: The camera extrinsics (rotation and translation).
  - `scene.frames[0].image.name`: The original source image name.
- `scene.summary`: A string summarizing the reconstruction statistics.

Existing models can be loaded in the same way:

```python
from pathlib import Path
from src.pipeline import run_load_pipeline

scene = run_load_pipeline(
    Path("./output/0").resolve(),
    Path("./images").resolve(),
)
```

For dense reconstruction, create a `DenseConfig` and call
`run_mvs_pipeline(config)`. It creates `dense/*.bin` to `config.mvs_dir` rather
than returning the fused point cloud in memory.

