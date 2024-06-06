import json
import argparse

parser = argparse.ArgumentParser(description='Makes list out of results JSON object')

parser.add_argument('--input', "-i", type=str)
parser.add_argument('--output', "-o", type=str)
parser.add_argument('--mincs', "-m", type=int)

args = parser.parse_args()

with open(args.input, "r") as file:
    results = [key for key, value in json.load(file)[0].items() if value >= args.mincs]

with open(args.output, "w") as file:
    for result in results:
        file.writelines(result + "\n")
