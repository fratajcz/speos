from speos.pipeline import TrainingPipeline
import argparse

parser = argparse.ArgumentParser(description='Run Training Pipeline.')

parser.add_argument('--config', "-c", type=str, default="config_big_bagging_linkx_mse.yaml",
                    help='Path to the config that should be used for the run.')

args = parser.parse_args()

pipeline = TrainingPipeline(args.config)

pipeline.run()
