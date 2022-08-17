import os
import hashlib
import time
import re


class Config(dict):
    def __init__(self, is_first=True, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

        if is_first:
            path = os.path.dirname(__file__)
            default_config_name = "config_default.yaml"
            self.parse_yaml(os.path.join(path, default_config_name), is_first=True)
            if self.name is None:
                self.name = self.get_hash()

            if self.timestamp is None:
                self.timestamp = time.time()

    def parse_yaml(self, yaml_file, is_first=False):
        import yaml
        with open(yaml_file, "r") as file:
            payload = yaml.load(file, Loader=yaml.SafeLoader)
            if is_first:
                self.recursive_create(payload)
            else:
                self.recursive_update(self, payload)
        self.__dict__ = self

    def save(self, path=None):
        import yaml

        if path is None:
            path = os.path.join(self.model.save_dir, str(self.name) + ".yaml")
        with open(path, 'w') as yaml_file:
            yaml.dump(self.recursive_serialize(self.__dict__), yaml_file, default_flow_style=False)

    def recursive_serialize(self, nondict):
        serializable_dict = {}
        for key, value in nondict.items():
            if value.__class__ == self.__class__:
                value = self.recursive_serialize(value)

            serializable_dict.update({key: value})
        return serializable_dict

    def recursive_create(self, payload):
        for key, value in payload.items():
            if type(value) == dict:
                subconfig = Config(is_first=False)
                subconfig.recursive_create(value)
                self.update({key: subconfig})
            else:
                if value == "None":
                    value = None
                elif type(value) == str and re.match(r"\d+e-\d+", value):
                    value = float(value)
                self.update({key: value})

    def recursive_update(self, default, payload):
        for (key, value) in list(default.items()):
            if key == "kwargs" and key in payload.keys():
                default[key].update(payload[key])
            elif type(value) == Config:
                if key in payload.keys():
                    self.recursive_update(value, payload[key])
            else:
                if key in payload.keys():
                    if payload[key] == "None":
                        payload[key] = None
                    default.update({key: payload[key]})
                    default.key = payload[key]

    def deepcopy(self):
        new_config = Config()
        self.recursive_update(new_config, self)
        return new_config

    @property
    def _hash(self):
        m = hashlib.blake2b(digest_size=3)
        m.update(str(self).encode("utf-8"))
        return m.hexdigest()

    def get_hash(self):
        return self._hash
