import yaml 

def read_yaml_file(path_to_file):
    try:
        with open(path_to_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print("Error reading the config file")

    
