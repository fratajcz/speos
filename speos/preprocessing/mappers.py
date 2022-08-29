import os
import json
import re
from speos.utils.logger import setup_logger

class Mapper:
    """Abstract Mapper Class"""

    def get_mappings(self, tags: str = "", fields: str = "name"):
        '''goes through the mapping list and returns all mappings that include the provided tag in the provided field (default is name field)

          If called without arguments, returns all mappings (tag = "") '''
        if type(tags) == str:
            tags = [tags]
        if type(fields) == str:
            fields = [fields]

        if len(tags) > len(fields) and len(fields) == 1:
            mappings = []
            for mapping in self.mapping_list:
                appendFlag = False
                for tag in tags:
                    if tag in mapping[fields[0]]:
                        appendFlag = True
                if appendFlag:
                    mappings.append(mapping)
        else:
            mappings = []
            for mapping in self.mapping_list:
                appendFlag = True
                for tag, field in zip(tags, fields):
                    if tag not in mapping[field]:
                        appendFlag = False
                if appendFlag:
                    mappings.append(mapping)

        return mappings


class GWASMapper(Mapper):
    r"""Handles the mapping of y labels to GWAS feature files.

    Enables simple matching of multiple GWAS to individual Phenotypes via its :obj:`get_mappings()` method.

    Args:
        ground_truth_path (int): The path to the directory where the ground truth (label) files are stored, as defined in the :obj:`mapping_file`.
            It can be set to :obj:`""` if the path is included in the file name as defined in the :obj:`mapping_file`.
        features_file_path (str): The path to the directory where the features (GWAS) files are stored, as defined in the :obj:`mapping_file`.
            It can be set to :obj:`""` if the path is included in the file name as defined in the :obj:`mapping_file`.
        mapping_file (str): The path to the file that maps ground truths (labels) to sets of feature (GWAS) files. (default: :obj:`./speos/mapping.json`)
    """
    def __init__(self,
                 ground_truth_path: str,
                 features_file_path: str,
                 mapping_file: str = "./speos/mapping.json",
                 extension_mappings: str = "./extensions/mapping.json"):

        self.features_file_path = features_file_path
        self.ground_truth_path = ground_truth_path

        self.mapping_list = []

        for mapping in mapping_file, extension_mappings:
            with open(mapping, "r") as file:
                content = file.read()
                self.mapping_list.extend(json.loads(content))

        mappings_to_delete = []
        for mapping in self.mapping_list:
            if mapping["features_file"] != "":
                mapping["ground_truth"] = os.path.join(
                    self.ground_truth_path, mapping["ground_truth"])
                mapping["features_file"] = os.path.join(
                    self.features_file_path, mapping["features_file"])
            else:
                mappings_to_delete.append(mapping)

        for mapping in mappings_to_delete:
            del self.mapping_list[self.mapping_list.index(mapping)]
            

class AdjacencyMapper(Mapper):
    r"""Handles the mapping of names and network types to their respective files.

    Enables simple matching of multiple Networks to individual queries via its :obj:`get_mappings()` method.

    Args:
        mapping_file (str): The path to the file that describes the networks and where they are stored. (default: :obj:`./speos/adjacencies.json`)
    """
    def __init__(self,
                 mapping_file: str = "speos/adjacencies.json",
                 extension_mappings: str = "./extensions/adjacencies.json"):

        self.mapping_list = []

        for mapping in mapping_file, extension_mappings:
            with open(mapping, "r") as file:
                content = file.read()
                self.mapping_list.extend(json.loads(content))

        mappings_to_delete = []
        for mapping in self.mapping_list:
            if mapping["file_path"] != "":
                continue
            else:
                mappings_to_delete.append(mapping)

        for mapping in mappings_to_delete:
            del self.mapping_list[self.mapping_list.index(mapping)]

        self._format_mapping_names()

    def _format_mapping_names(self):
        for mapping in self.mapping_list:
            # get rid of whitespaces, punctuation and special characters
            mapping["name"] = self._format_name(mapping["name"])

    def _format_name(self, string):
        return re.sub('[^A-Za-z0-9]+', '', string)

    def get_mappings(self, tags: str = "", fields: str = "name"):
        '''goes through the mapping list and returns all mappings that include the provided tag in the provided field (default is name field)

          If called without arguments, returns all mappings (tag = "") '''
        if type(tags) == str:
            tags = [tags]
        if type(fields) == str:
            fields = [fields]

        assert len(tags) == len(fields) or len(fields) == 1

        for i in range(len(tags)):
            try:
                if fields[i] == "name":
                    tags[i] = self._format_name(tags[i])
            except IndexError:
                if fields[0] == "name":
                    tags[i] = self._format_name(tags[i])

        return super().get_mappings(tags, fields)

