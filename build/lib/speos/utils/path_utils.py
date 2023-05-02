import os


def processed_data_dir(config) -> str:
    """ Returns the processed data dir """
    return os.path.join(config.input.save_dir, "processed")


def processed_data_filename(config) -> tuple:
    """ Returns the processed data filenames for .pt and .tsv file """
    return [config.name + "{}".format(ending) for ending in [".pt", ".tsv"]]


def processed_data_path(config) -> tuple:
    """ Returns the processed data paths for .pt and .tsv file) """
    return [os.path.join(processed_data_dir(config), filename) for filename in processed_data_filename(config)]


def postprocessing_results_path(config) -> str:
    """ Returns the path where postprocessing results should be saved """
    return config.pp.save_dir


def postprocessing_plots_path(config) -> str:
    """ Returns the path where postprocessing plots should be saved """
    return config.pp.plot_dir