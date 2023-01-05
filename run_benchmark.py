from speos.benchmark import TestBench
import argparse

parser = argparse.ArgumentParser(description='Run Crossvalidation Inference Pipeline.')

parser.add_argument('--config', "-c", type=str, default="",
                    help='Path to the config that should be used for the run.')
parser.add_argument('--parameters', "-p", type=str,
                    help='Path to the parameter list that should be assessed.')

args = parser.parse_args()

tb = TestBench(parameter_file=args.parameters, config_path=args.config)

tb.run()
