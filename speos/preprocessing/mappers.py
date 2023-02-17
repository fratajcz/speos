import json
import re


class Mapper:
    """Abstract Mapper Class"""
    def __init__(self, blacklist: list = []):
        self.blacklist = blacklist

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

        return self.remove_blacklisted_mappings(tags, mappings)

    def remove_blacklisted_mappings(self, tags, mappings):
        blacklisted_mappings = []
        for banned_mapping in self.blacklist:
            for tag in tags:
                # skip removing blacklisted adjacencies if they are explicitely requested
                if banned_mapping.lower() in tag.lower():
                    skip_mapping = True
                else: 
                    skip_mapping = False
            if skip_mapping:
                continue

            for mapping in mappings:
                if banned_mapping in mapping["name"].lower():
                    blacklisted_mappings.append(mapping)

        for banned_mapping in blacklisted_mappings:
            mappings.remove(banned_mapping)

        return mappings


class GWASMapper(Mapper):
    r"""Handles the mapping of y labels to GWAS feature files.

    Enables simple matching of multiple GWAS to individual Phenotypes via its :obj:`get_mappings()` method.

    Args:
        mapping_file (str): The path to the file that maps ground truths (labels) to sets of feature (GWAS) files. (default: :obj:`./speos/mapping.json`)
        extension_mappings (str): The path to the file that maps ground truths (labels) to sets of feature (GWAS) files for user-defined extensions. (default: :obj:`./extensions/mapping.json`)
        
    """
    def __init__(self,
                 mapping_file: str = "./speos/mapping.json",
                 extension_mappings: str = "./extensions/mapping.json",
                 **kwargs):
        super().__init__(**kwargs)

        self.mapping_list = []
        self.backup_mapping = None

        for mapping in [mapping_file, extension_mappings]:
            with open(mapping, "r") as file:
                content = file.read()
                self.mapping_list.extend(json.loads(content))

        mappings_to_delete = []
        for mapping in self.mapping_list:
            if mapping["features_file"] == "":
                mappings_to_delete.append(mapping)
                self.backup_mapping = mapping

        for mapping in mappings_to_delete:
            del self.mapping_list[self.mapping_list.index(mapping)]

    def get_mappings(self, *args, **kwargs):
        """ Returns mappings fitting the description. If the description returns no mappings due to missing GWAS files, just return one of them so we have the mapping to the labels """

        mappings = super().get_mappings(*args, **kwargs)
        if len(mappings) == 0:
            mappings = [self.backup_mapping]
        return mappings
            

class AdjacencyMapper(Mapper):
    r"""Handles the mapping of names and network types to their respective files.

    Enables simple matching of multiple Networks to individual queries via its :obj:`get_mappings()` method.

    Args:
        mapping_file (str): The path to the file that describes the networks and where they are stored. (default: :obj:`./speos/adjacencies.json`)
        extension_mappings (str): The path to the file that describes the networks and where they are stored for user-defined extensions. (default: :obj:`./extensions/adjacencies.json`)
    """
    def __init__(self,
                 mapping_file: str = "speos/adjacencies.json",
                 extension_mappings: str = "./extensions/adjacencies.json",
                 **kwargs):
        super().__init__(**kwargs)

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

        If called without arguments, returns all mappings (tag = "") 
          
        Args:
            tags (str/list): the tag or a list of tags that should be searched for in the fiven field of adjacencies (i.e. a name, a type etc.) 
            fields (str/list): the field in which the tag should be searched for. in case field is a string, all tags are searched in that field.
                In case of multiple tags and multiple fields, lengths must match and the nth tag is searched in the nth field.

        Returns:
            list: List of adjacencies that match the tag/field mapping and are not blacklisted.
          '''
        if type(tags) == str:
            tags = [tags]
        if type(fields) == str:
            fields = [fields]

        tags = ["" if tag == "all" else tag for tag in tags]

        assert len(tags) == len(fields) or len(fields) == 1

        for i in range(len(tags)):
            try:
                if fields[i] == "name":
                    tags[i] = self._format_name(tags[i])
            except IndexError:
                if fields[0] == "name":
                    tags[i] = self._format_name(tags[i])

        return super().get_mappings(tags, fields)


