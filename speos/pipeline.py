from speos.experiment import Experiment, InferenceEngine
from speos.wrappers import CVWrapper, OuterCVWrapper, BaggingWrapper
from speos.postprocessing.postprocessor import PostProcessor
from speos.utils.config import Config
from speos.utils.logger import setup_logger


class Pipeline:
    def __init__(self, config_path: str = ""):
        """Abstract class for Pipelines"""
        self.config = Config()
        self.logger_args = self.config, __name__

        if config_path != "":
            self.config.parse_yaml(config_path)
            logger = setup_logger(*self.logger_args)
            try:
                self.config.save()
            except FileNotFoundError:
                logger.warning("Failed saving config. Please check write permissions.")
        else:
            logger = setup_logger(*self.logger_args)
        # here the necessary modules for the pipeline should be initialized

    def run(self):
        # and here the modules should be run
        raise NotImplementedError


class TrainingPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(TrainingPipeline, self).__init__(config_path)

        self.experiment = Experiment(self.config)
        self.inference_engine = InferenceEngine(self.config)
        #self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.experiment.run()
        self.inference_engine.infer()
        #self.postprocessor.run()


class CVPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(CVPipeline, self).__init__(config_path)

        self.crossval = CVWrapper(self.config, self.logger)
        self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.crossval.run()
        self.postprocessor.run()


class BaggingPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(BaggingPipeline, self).__init__(config_path)

        self.bagging = BaggingWrapper(self.config, self.logger)
        # self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.bagging.run()
        # TODO: make postprocessor bagging-ready
        # self.postprocessor.run()


class OuterCVPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(OuterCVPipeline, self).__init__(config_path)

        self.crossval = OuterCVWrapper(self.config, self.logger)
        self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.crossval.run()
        self.postprocessor.run()


class InferencePipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(InferencePipeline, self).__init__(config_path)

        self.inference_engine = InferenceEngine(self.config, self.logger)
        #self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.inference_engine.infer()
        #self.postprocessor.run()


class CVTrainingPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(CVInferencePipeline, self).__init__(config_path)
        if self.config.pp.switch:
            logger = setup_logger(*self.logger_args)
            logger.info("Found train switch is 'on' in the provided config, turning it off to do inference-only.")
            self.config.pp.switch = False

        self.crossval = CVWrapper(self.config)

    def run(self):
        self.crossval.run()


class CVInferencePipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(CVInferencePipeline, self).__init__(config_path)
        if self.config.training.switch:
            logger = setup_logger(*self.logger_args)
            logger.info("Found train switch is 'on' in the provided config, turning it off to do inference-only.")
            self.config.training.switch = False

        self.crossval = CVWrapper(self.config, self.logger)
        self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.crossval.run()
        self.postprocessor.run()


class OuterCVInferencePipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(OuterCVInferencePipeline, self).__init__(config_path)
        if self.config.training.switch:
            logger = setup_logger(*self.logger_args)
            logger.info("Found train switch is 'on' in the provided config, turning it off to do inference-only.")
            self.config.training.switch = False

        self.crossval = OuterCVWrapper(self.config, self.logger)
        self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.crossval.run()
        self.postprocessor.run()


class PostProcessPipeline(Pipeline):
    def __init__(self, config_path: str = ""):
        super(PostProcessPipeline, self).__init__(config_path)
        logger = setup_logger(*self.logger_args)
        if self.config.training.switch:
            logger.info("Found train switch is 'on' in the provided config, turning it off to do postprocessing-only.")
            self.config.training.switch = False

        if self.config.inference.switch:
            logger.info("Found inference switch is 'on' in the provided config, turning it off to do postprocessing-only.")
            self.config.inference.switch = False
        self.postprocessor = PostProcessor(self.config)

    def run(self):
        self.postprocessor.run()
