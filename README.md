# olist_raw_data_API

This code is a FastAPI application for uploading, listing and transferring .csv files to Google Cloud Storage.

The create_upload_file endpoint is a POST endpoint for uploading a .csv file via a multipart/form-data POST request. The file is stored in a folder named 'data' on the server. The uploaded file's path is returned in the JSON response.

The list_files endpoint is a GET endpoint that returns the names of all the files in the 'data' folder in a JSON response.

The upload_to_gcs endpoint is a GET endpoint that uploads a single file to Google Cloud Storage. It takes in the name of the file, the Google Cloud project name and the bucket name as parameters and returns a message indicating the successful upload of the file.

The upload_all_files endpoint is a GET endpoint that uploads all the files in the 'data' folder to Google Cloud Storage. It takes in the Google Cloud project name and the bucket name as parameters and returns a message indicating the completion of the upload.