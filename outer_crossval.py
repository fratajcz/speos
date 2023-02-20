from speos.pipeline import OuterCVPipeline
import argparse

parser = argparse.ArgumentParser(description='Run Outer Crossvalidation Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

pipeline = OuterCVPipeline(args.config)
pipeline.run()