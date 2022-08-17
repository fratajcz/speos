from speos.pipeline import *
import argparse

parser = argparse.ArgumentParser(description='Run Crossvalidation Inference Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

#pipeline = CVInferencePipeline("inference_config.yaml")

pipeline = CVInferencePipeline(args.config)


pipeline.run()
