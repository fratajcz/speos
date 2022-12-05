from os import listdir
from os.path import isfile, join

import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

parser = argparse.ArgumentParser(description='Binds all tsv which share a given string together into one xlsx Workbook')

parser.add_argument('--dir', "-d", type=str, default=".",
                    help='Directory where Files are located (and output will be placed).')
parser.add_argument('--string', "-s", type=str,
                    help='Match by this string.')

args = parser.parse_args()

onlyfiles = [f for f in listdir(args.dir) if isfile(join(args.dir, f)) and args.string in f and ".xlsx" not in f]

print("Gathering files: {}".format(onlyfiles))

filename = args.string + ".xlsx"
print("Into file: {}".format(filename))

workbook = Workbook()
main_sheet = workbook.active
main_sheet.title = "Header"
first = True
for file in onlyfiles:
    df = pd.read_csv(join(args.dir, file), sep="\t")
    if first:
        # paste the column header terms onto the first sheet
        for i in range(5):
            main_sheet.append([])
        for term in df.columns:
            main_sheet.append([term])
        first = False
    without_ending = file.split(".")[0]
    unique_part = "".join(without_ending.split(args.string))
    new_sheet = workbook.create_sheet(unique_part[:30])
    for row in dataframe_to_rows(df, index=False, header=True):
        new_sheet.append(row)

workbook._sheets.sort(key=lambda ws: ws.title)

workbook.save(join(args.dir, filename))