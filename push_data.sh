aws --endpoint-url http://localhost:4566/ s3 mb s3://document-classification-2024
aws --endpoint-url http://localhost:4566/ s3 sync ./data-original s3://document-classification-2024/data