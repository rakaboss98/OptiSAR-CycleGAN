import os
import boto3
from utils import readyaml

# Configure access to the S3 bucket using credential

path_name = "aws.yaml"
config = readyaml.read_yaml_file(path_name)
access_key=config["CREDENTIALS"]["ACCESS_KEY"]
secret_access_key=config["CREDENTIALS"]["SECRET_ACCESS_KEY"]

# Configure boto3

client = boto3.client('s3',
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_access_key)

data_dir_path = config["TRAINING_DATA"]["PATH"]
bucket_name = config["TRAINING_DATA"]["BUCKET_NAME"]
pth_in_bucket = config["TRAINING_DATA"]["PATH_IN_BUCKET"]

# The path should strictly contain only directories and not any file
def uploadDirectory(path, bucket_name, pth_in_bucket):
    for root, dirs, files in os.walk(path):
        path = os.path.dirname(root+"/")
        base_name = os.path.basename(path)

        for file in files:
            client.upload_file(os.path.join(root, file),
                               bucket_name,
                               pth_in_bucket + base_name + "/" + str(file))


uploadDirectory(data_dir_path, bucket_name, pth_in_bucket)
