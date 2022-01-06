from barusini.nn.generic.utils import parse_config
import os
import glob


class Serializable:
    @staticmethod
    def get_config_path(folder_path):
        return os.path.join(folder_path, "high_level_config.json")

    @classmethod
    def from_folder(cls, folder_path=None, **overrides):  # load fitted
        config_path = Serializable.get_config_path(folder_path)
        return cls.from_config(config_path, **overrides)

    @classmethod
    def from_config(cls, config_path, **overrides):  # load unfitted
        # add reference to original config file to the overrides
        if "original_config_path" not in overrides:
            overrides["original_config_path"] = config_path

        config = parse_config(config_path, **overrides)
        return cls(**config)

    def to_folder(self, folder_path=None):
        pass

    @staticmethod
    def find_best_ckpt(file):
        path = os.path.join(file, "epoch=*.ckpt")
        all_possible_ckpts = glob.glob(path)
        assert len(all_possible_ckpts) > 0, f"No file matching {path} found"
        assert (
            len(all_possible_ckpts) < 2
        ), f"More files matching {path} found: {all_possible_ckpts}"
        return all_possible_ckpts[0]
