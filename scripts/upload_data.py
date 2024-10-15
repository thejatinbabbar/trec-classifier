import os
import boto3

s3 = boto3.client('s3', endpoint_url='http://localstack:4566', aws_access_key_id='localstack', aws_secret_access_key='localstack', region_name='us-east-1')

bucket_name = 'document-classification-2024'
folder_path = "/data-original"
s3_prefix = "data"

# Create a bucket in LocalStack
s3.create_bucket(Bucket=bucket_name)

for root, _, files in os.walk(folder_path):
    for file in files:
        # Local file path
        local_file_path = os.path.join(root, file)

        # Compute S3 object key
        relative_path = os.path.relpath(local_file_path, folder_path)
        s3_key = os.path.join(s3_prefix, relative_path)  # .replace('\\', '/')

        # Upload the file
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f'Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}')
