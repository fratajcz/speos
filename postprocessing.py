from speos.pipeline import PostProcessPipeline
import argparse

parser = argparse.ArgumentParser(description='Run Post Processing Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

pipeline = PostProcessPipeline(args.config)

#pipeline = PostProcessPipeline("c7e39d.yaml")

pipeline.run()