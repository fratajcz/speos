from speos.pipeline import CVInferencePipeline
import argparse

parser = argparse.ArgumentParser(description='Run Crossvalidation Inference Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

pipeline = CVInferencePipeline(args.config)


pipeline.run()
