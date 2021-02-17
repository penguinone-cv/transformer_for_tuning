from train import Trainer
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="main file for MNIST classification Transformer")
    parser.add_argument("yaml_name", help="setting yaml file name")
    args = parser.parse_args()

    yaml_file = args.yaml_name + ".yaml"
    base_parameters_dir = "./parameters"

    setting_yaml_file = os.path.join(base_parameters_dir, yaml_file)
    trainer = Trainer(setting_yaml_file=setting_yaml_file)
    if not os.path.isfile(os.path.join(trainer.parameters.log_path, trainer.parameters.model_name)):
        print("Trained weight file does not exist")
        trainer.train()

if __name__ == '__main__':
    main()
