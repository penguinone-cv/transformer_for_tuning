import subprocess
import json
import os
import yaml

class Parameters:
    def __init__(self, setting_yaml_file):
        all_parameters = self.read_yaml(setting_yaml_file)
        self.model_name = all_parameters["model_name"]
        self.data_path = all_parameters["data_path"]
        self.fixed_parameters = all_parameters["fixed_parameters"]
        self.model_parameters = all_parameters["model_parameters"]
        self.learning_parameters = all_parameters["learning_parameters"]
        self.img_size = (self.fixed_parameters["width"], self.fixed_parameters["height"])
        log_dir_name = self.model_name + "_layers" + str(self.model_parameters["layers_num"]) + \
                        "_heads" + str(self.model_parameters["heads"]) + \
                        "_hidden" + str(self.model_parameters["hidden_dim"]) + \
                        "_tokens" + str(self.model_parameters["tokens"]) + \
                        "_epochs" + str(self.learning_parameters["epochs"]) + \
                        "_batch_size" + str(self.learning_parameters["batch_size"]) + \
                        "_lr" + str(self.learning_parameters["learning_rate"]) + \
                        "_lrdecay" + str(self.learning_parameters["lr_decay"])
        self.log_path = os.path.join(all_parameters["base_log_path"], log_dir_name)

        #self.fixed_parameters["pin_memory"] = self.str_to_bool(self.fixed_parameters["pin_memory"])
        if self.fixed_parameters["num_workers"] == -1:
            self.fixed_parameters["num_workers"] = os.cpu_count()

    def read_yaml(self, setting_yaml_file):
        with open(setting_yaml_file) as f:
            return yaml.safe_load(f)

    #文字列のTrueをbool値のTrueに変換しそれ以外をFalseに変換する関数
    def str_to_bool(self, str):
        return str.lower() == "true"
