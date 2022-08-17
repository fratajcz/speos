from speos.pipeline import CVPipeline
import argparse

parser = argparse.ArgumentParser(description='Run Crossvalidation Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

pipeline = CVPipeline(args.config)
pipeline.run()