import csv
import json
import os
import shutil
from pathlib import Path
import os
import boto3
from concurrent.futures import ThreadPoolExecutor

# Function to read metadata definition from JSON file
def read_metadata_definition(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata_definition = json.load(f)
    return metadata_definition

# Function to process CSV file and generate output files
def process_csv(input_csv, metadata_definition,data_directory):
    embedding_attributes = metadata_definition['csv']['embeddingattributes']
    metadata_attributes = metadata_definition['csv']['metadataattributes']
    index_id_attributes = metadata_definition['csv']['index_id']

    # Create or clean outputs folder within data folder
    data_dir = data_directory
    outputs_dir = data_directory + "/outputs_1k"

    
    
    with open(input_csv, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            
            # Generate index_id based on attributes provided
            index_id_values = [row[attr] for attr in index_id_attributes]
            index_id = '_'.join(index_id_values)

            # Process embedding attributes
            embedding_data = {attr: row[attr] for attr in embedding_attributes}
            embedding_output_path = os.path.join(outputs_dir, f'{index_id}.csv')
            with open(embedding_output_path, 'w', newline='') as embedding_csv:
                csv_writer = csv.DictWriter(embedding_csv, fieldnames=embedding_attributes)
                csv_writer.writeheader()
                csv_writer.writerow(embedding_data)

            # Process metadata attributes
            metadata_data = {attr: row[attr] for attr in metadata_attributes}
            metadata_output_path = f'{embedding_output_path}.metadata.json'
            with open(metadata_output_path, 'w') as metadata_json:
                json.dump({'metadataAttributes': metadata_data}, metadata_json, indent=4)




def upload_file(bucket_name, local_path, s3_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, bucket_name, s3_path)
    print(f'Uploaded {local_path} to s3://{bucket_name}/{s3_path}')

def upload_files_to_s3(bucket_name, local_directory, s3_prefix=''):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    with ThreadPoolExecutor(max_workers=500) as executor:  # Adjust max_workers as needed
        futures = []
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.join(s3_prefix, os.path.relpath(local_path, local_directory))
                futures.append(executor.submit(upload_file, bucket_name, local_path, s3_path))
        
        for future in futures:
            future.result()

# Main function
def main():
    bucket_name = 'kaggle-1k-courses'
    local_directory = '/Users/anshumaantyagi/Desktop/CourseRec/course_matching/data/outputs_1k'
    data_directory = '/Users/anshumaantyagi/Desktop/CourseRec/course_matching/data'  
    input_csv = '/Users/anshumaantyagi/Desktop/CourseRec/course_matching/data/1k_courses.csv'
    metadata_file = '/Users/anshumaantyagi/Desktop/CourseRec/course_matching/data/knowledgebasemetadataprep.json'
    metadata_definition = read_metadata_definition(metadata_file)
    print(metadata_definition)
    print("Prepping CSV....")
    process_csv(input_csv, metadata_definition, data_directory)
    print(f"Uploading to {bucket_name}")
    upload_files_to_s3(bucket_name, local_directory)

if __name__ == '__main__':
    main()