from speos.preprocessing.mappers import *
from speos.preprocessing.preprocessor import PreProcessor

class InputHandler:
    def __init__(self, config, **preprocessor_kwargs):
        """ Utility class that strings together gwas and adjacency mapping and feeds it into the preprocessor 

            Args:
                config (speos.utils.config.Config): A Config object that contains all the details of the required data.
                **preprocessor_kwargs (dict): Keyword arguments that are handed down to the preprocessor

        """
        self.config = config
        mappings = GWASMapper().get_mappings(config.input.tag, fields=config.input.field)

        adjacencies = AdjacencyMapper(config.input.adjacency_mappings, blacklist=self.config.input.adjacency_blacklist).get_mappings(config.input.adjacency, fields=config.input.adjacency_field)

        self.prepro = PreProcessor(config, mappings, adjacencies, **preprocessor_kwargs)

    def get_preprocessor(self):
        """ 
            Returns:
                speos.preprocessing.preprocessor.Preprocessor: The Preprocessor object that holds all the data necessary for the run in graph format.
        """
        return self.prepro

    def get_data(self, *args, **kwargs):
        """ Utility function that calls get_data of the preprocessor

            Returns:
                tuple(Tensor, Tensor, Tensor): returns input matrix X, ground truth y and adjacency matrix adj as pytorch tensors.
        """
        return self.prepro.get_data(*args, **kwargs)