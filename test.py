import yaml

with open('parameters/test.yaml') as f:
    dict = yaml.safe_load(f)
    print(dict)
