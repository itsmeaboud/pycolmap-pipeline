import logging
import argparse
from ast import arg, parse
from pathlib import Path
from src import sparse
from src.pipeline import run_load_pipeline, run_sparse_pipeline, run_load_reconstruction, run_mvs_pipeline
from src.data_types import CameraModelType, MatcherType, ExtractorType, SparseConfig, DenseConfig
from src.vis_util import create_rrd, stream_data
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)



def main():
    
    parser  = argparse.ArgumentParser(description = "COLMAP Pipeline CLI")
    subparser = parser.add_subparsers(dest = "command", required = True, help = "pipline to run") 
    
    
    # Sparse pipeline
    sparse_parser = subparser.add_parser(
        "sparse",
        help = "Run sparse reconstruction pipeline (feature extraction and matching"
    )
    
    sparse_parser.add_argument(
        "images_dir",
        type = str,
        help = "Path to directory containing images",    
    )
    
    sparse_parser.add_argument(
        "output_dir",
        type = str,
        help = "Path to save output workspace and reconstruction files"
    )
    
    sparse_parser.add_argument(
        "--database", "--db",
        type = str,
        default = "database.db",
        help = "Database file name (default: database.db)",
    )
    
    sparse_parser.add_argument(
        "--camera-model",
        type = str,
        choices = [model.value for model in CameraModelType],
        default = CameraModelType.SIMPLE_RADIAL.value,
        help = "Camera projection model type (default: SIMPLE_RADIAL)",
    )
    
    sparse_parser.add_argument(
        "--extractor", "-e",
        type = str,
        choices = [extractor.value for extractor in ExtractorType],
        default = "SIFT",
        help = "Feature extraction to use (default: SIFT)",
    )
    
    sparse_parser.add_argument(
        "--visualize",
        action = "store_true",
        help = "Launch Rerun viewer"
    )
    
    sparse_parser.add_argument(
        "--matcher", "-m",
        type = str,
        choices = [matcher.value for matcher in MatcherType],
        default = MatcherType.EXHAUSTIVE.value,
        help = "Feature matching to use (default: EXHAUSTIVE)"
    )
    
    
    # MVS pipeline
    dense_parser = subparser.add_parser(
        "mvs",
        help = "Run MVS reconstruction pipeline"
        )
    
    dense_parser.add_argument(
        "images_dir",
        type = str,
        default = "Path to directory containing images"
        )
    dense_parser.add_argument(
        "output_dir",
        type = str,
        default = "Path to directory containing sparse reconstruction (.bin)"
        )
    
    dense_parser.add_argument(
        "mvs_dir",
        type = str,
        help = "Path to store MVS reconstruction output"
        )
    
    # Load pipeline
    load_parser = subparser.add_parser(
        "load",
        help = 'Load existing sparse/dense project (.bin)'
        )
    load_parser.add_argument(
        "images_dir",
        type = str,
        help = ""
    )    
    load_parser.add_argument(
        "model",
        type = str,
        help = "Path to dense/sparse project (.bin) "
        )
    

    
    

    args = parser.parse_args()
    
    if args.command == "sparse":
        
        cfg = SparseConfig(
            images_dir = Path(args.images_dir).resolve(),
            output_dir = Path(args.output_dir).resolve(),
            db_dir = (Path(args.output_dir) / args.database).resolve(),
            camera_model = CameraModelType(args.camera_model),
            matcher_type = MatcherType(args.matcher),
            extractor_type = ExtractorType(args.extractor)
        )
    
        reconstruction_output = run_sparse_pipeline(cfg)
        
        if args.visualize:
            
            create_rrd(reconstruction_output, Path(args.output_dir))
        
    
    if args.command == "mvs":
        
        cfg = DenseConfig(
            images_dir = Path(args.images_dir).resolve(),
            output_dir = Path(args.output_dir).resolve(),
            mvs_dir = Path(args.mvs_dir).resolve(),
        )
        
        run_mvs_pipeline(cfg)
        
    if args.command == "load":
        
        model_path = Path(args.model).resolve()
        
        reconstruction_output = run_load_pipeline(model_path, Path(args.images_dir))
        stream_data(reconstruction_output)
        
        

    
if __name__ == "__main__":
    
    main()
    
    
