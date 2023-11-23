import yaml

from lutils.dict_wrapper import DictWrapper


class Configuration(DictWrapper):
    """
    Represents the configuration parameters for running the process
    """

    def __init__(self, path: str):
        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super(Configuration, self).__init__(config)

        self.check_config()

    def check_config(self):
        if "max_num_videos" not in self["data"]:
            self["data"]["max_num_videos"] = None

        if "offset" not in self["data"]:
            self["data"]["offset"] = None

        if "reference" not in self["model"]:
            self["model"]["reference"] = True

        if "past_horizon" not in self["evaluation"]:
            self["evaluation"]["past_horizon"] = -1

        if "warm_sampling" not in self["evaluation"]:
            self["evaluation"]["warm_sampling"] = 0.0

        if "steps" not in self["evaluation"]:
            self["evaluation"]["steps"] = 100
