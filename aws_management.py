import os
import boto3

# Configure access to the S3 bucket using credentials

access_key = 'AKIAQVSA7EXYBCJRHLR7'
secret_access_key = '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'

# Configure boto3

client = boto3.client('s3',
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_access_key)

data_dir_path = "/Users/rakshitbhatt/Documents/GalaxEye /Datasets/archive"

bucket_name = 'sample-datasets-galaxeye'
pth_in_bucket = 'Training Datasets/'


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
