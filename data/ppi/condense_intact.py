import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Condenses Intact Download to Ensembl gene IDs A and B')

parser.add_argument('--input', "-i", type=str, default="/home/icb/florin.ratajczak/ppi-core-genes/data/ppi/intact.txt",
                    help='Path to the raw intact file (can be zipped)')
parser.add_argument('--key', "-k", type=str, default="ensembl:ENSP",
                    help='Key that has to be present in both A and B')            

args = parser.parse_args()

physical_lines = []
direct_lines = []

# identifiers according to https://www.ebi.ac.uk/ols/ontologies/MI/terms?obo_id=MI:0190
direct_identifiers = ["direct interaction", "covalent binding", "disulfide bond"]
physical_identifier = "physical association"

with open(args.input, "r") as file:
    for i, line in enumerate(tqdm(file)):
        if i == 0:
            # skip header
            continue
        cells = line.split("\t")
        A_names = cells[2]
        B_names = cells[3]
        interactions = cells[11]
        A_name = False
        B_name = False

        if args.key in A_names and args.key in B_names:
            A_split = A_names.split("|")
            B_split = B_names.split("|")
            for name in A_split:
                if args.key in name:
                    A_name = name
                    break

            for name in B_split:
                if args.key in name:
                    B_name = name
                    break

            if A_name and B_name:
                if physical_identifier in interactions:
                    physical_lines.append("{}\t{}\n".format(A_name, B_name))
                for identifier in direct_identifiers:
                    if identifier in interactions:
                        physical_lines.append("{}\t{}\n".format(A_name, B_name))
                        direct_lines.append("{}\t{}\n".format(A_name, B_name))

filenames = [args.input.split(".")[0] + "_{}.txt".format(interaction_type) for interaction_type in ["pa", "direct"]]

for filename, typed_list in zip(filenames, [physical_lines, direct_lines]):
    with open(filename, "w") as file:
        file.writelines(["GeneA\tGeneB\n"] + list(set(typed_list)))
