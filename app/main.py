# Import the necessary libraries for the application
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse
from google.cloud import storage
import modules.ETL_procedures as ETL
import modules.recomendation_procedures as MR
import modules.NLP_procedures as NLP
import pandas as pd
from apyori import apriori
import pickle
import os
import google.auth



# Instance FastAPI to create a new web application
app = FastAPI()

@app.get('/')
async def redirect():
    return RedirectResponse(url='/docs')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    """
    Endpoint for uploading a .csv file.

    This endpoint allows for the upload of a .csv file via a `multipart/form-data` POST request. 
    The file is stored in a folder named 'data' on the server.

    Parameters:
        file (UploadFile): The .csv file to be uploaded.

    Returns:
        JSON: A JSON response containing the file path of the uploaded file.
    """
    folder = "data/datasets"
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"file_path": file_path}


@app.get("/files/")
def list_files():
    """
    Endpoint to list all the files in the 'data' folder.

    Returns:
        JSON: A JSON response containing the names of all the files in the 'data' folder.
    """
    folder = "data"
    file_list = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            file_list.append(os.path.join(root, d))
        for f in files:
            file_list.append(os.path.join(root, f))

    return {'files': file_list}


@app.get('/drop_file/{file_name}')
def drop_file(file_name: str):

    if os.path.exists(f'data/{file_name}'):
        os.remove(f"data/{file_name}")
    else:
        return {'ERROR': 'File no found'}

    return {'msh': 'File removed'}


@app.get('/drop_all_files/')
def drop_file():

    for file_name in os.listdir('data/'):

        file = 'data/' + file_name
        if os.path.isfile(file):
            print('Deleting file: ', file)
            os.remove(file)
    return {'msg': 'Files removed'}


@app.get('/upload_google_storage/{file_name},{project},{bucket}')
def upload_to_gcs(file_name: str, project: str, bucket: str):
    """
    Endpoint for uploading a file to Google Cloud Storage.

    This endpoint allows for the upload of a file to Google Cloud Storage. 
    The uploaded file will be stored in a folder named 'data' in the specified bucket.

    Parameters:
        file_name (str): The name of the file to be uploaded.
        project (str): The name of the Google Cloud project.
        bucket (str): The name of the Google Cloud Storage bucket.

    Returns:
        JSON: A JSON response indicating the location of the uploaded file.
    """
    # Create a client object to interact with the Google Cloud Storage API
    client = storage.Client(project=project)

    # Get a reference to the desired bucket
    bucket = client.bucket(bucket)

    # Create a new blob (representing the uploaded file) in the bucket
    blob = bucket.blob(file_name)

    # Upload the file to the blob
    blob.upload_from_filename(f"data/datasets/{file_name}")

    # Return a JSON response indicating the location of the uploaded file
    return {'msg': f"File {file_name} has been uploaded to gs://{bucket.name}/{blob.name}"}


@app.get('/upload_all_files_GCP/{project},{bucket}')
def upload_all_files(project: str, bucket: str):
    """
    Endpoint to upload all files in a folder to Google Cloud Storage.

    This endpoint takes in a Google Cloud Storage project name and a bucket name as arguments and 
    uploads all files in the `data` folder to the designated bucket.

    Parameters:
        project (str): The name of the Google Cloud Storage project.
        bucket (str): The name of the Google Cloud Storage bucket.

    Returns:
        JSON: A JSON response containing the message 'done', indicating that the files have been uploaded.
    """
    # Create a client to access the Google Cloud Storage API
    client = storage.Client(project=project)

    # Get a reference to the desired bucket
    bucket = client.bucket(bucket)

    # Iterate over all files in the 'data' folder
    for path in os.listdir('data/'):
        # Check if the current path is a file
        if os.path.isfile(os.path.join('data/', path)):
            try:
                # Upload the file to the bucket
                blob = bucket.blob(f"{path}")
                blob.upload_from_filename(f"data/{path}")
            except:
                # Continue with the next file if the current one could not be uploaded
                continue

    # Return the 'done' message
    return {'msg': 'done'}


@app.get('/execute_etl/')
def execute_etl():
    client = storage.Client(project='olist-project-374912')

    input_bucket = client.bucket('olist-project-374912-datasets')
    output_bucket = client.bucket('olist-project-374912-datawarehouse')

    ETL.etl_orders_payment(input_bucket=input_bucket,
                           output_bucket=output_bucket)
    ETL.etl_qualified_leads(input_bucket=input_bucket,
                            output_bucket=output_bucket)
    ETL.etl_cltv(input_bucket=input_bucket, output_bucket=output_bucket)
    # ETL.Facu_ETL(input_bucket=input_bucket, output_bucket=output_bucket)
    ETL.etl_closed_deals(input_bucket=input_bucket,
                         output_bucket=output_bucket)
    ETL.etl_PODTCWTLM(input_bucket=output_bucket, output_bucket=output_bucket)
    ETL.etl_CGRATMCCTTPM(input_bucket=input_bucket,
                         output_bucket=output_bucket)
    ETL.etl_geolocation(input_bucket=input_bucket, output_bucket=output_bucket)
    ETL.etl_MAPOPBCWLM(input_bucket=output_bucket, output_bucket=output_bucket)

    return {'msg': 'all done'}


@app.get('/MR_etl/')
def MR_etl():

    path = 'data/datasets'

    order_items = pd.read_csv(f"{path}/olist_order_items_dataset.csv")
    products = pd.read_csv(f"{path}/olist_products_dataset.csv")
    order_reviews = pd.read_csv(f"{path}/olist_order_reviews_dataset.csv")
    orders = pd.read_csv(f"{path}/olist_orders_dataset.csv")

    df1 = pd.merge(order_items, products, on='product_id', how='outer')
    df2 = pd.merge(df1, order_reviews, on='order_id', how='outer')
    df3 = pd.merge(df2, orders, on='order_id', how='outer')
    columns = ["product_id", "product_category_name",
               "price", "review_score", "customer_id", 'order_id']
    data = df3[columns]
    data = data.dropna()
    df1 = data.groupby('order_id')['product_id'].apply(
        list).reset_index(name='compras')
    compras = list(df1['compras'])
    association_rules = apriori(
        compras, min_support=0.00003, min_confidence=0.1, min_lift=1, min_length=2)
    dfUsuarioReviews = data.groupby(
        ['customer_id']).size().reset_index(name='reviews')
    usuarios = list(
        dfUsuarioReviews[dfUsuarioReviews['reviews'] > 1]['customer_id'])

    # Crea DataFrame seleccionando de dfConsolidado los usuarios que tengan más de 1 review
    dfUsuarioProducto = data[data['customer_id'].isin(usuarios)]

    # Crea la matriz de review_score usuarios_productos_matrix
    usuarios_productos_matrix = dfUsuarioProducto.pivot_table(
        values='review_score', index='customer_id', columns='product_id')

    # Rellena los faltantes de la matriz con 0
    usuarios_productos_matrix = usuarios_productos_matrix.fillna(0)

    # Guardar el matrix de usuario - producto
    with open('data/recomendation/usuarios_productos_matrix.pickle', 'wb') as file:
        pickle.dump(usuarios_productos_matrix, file)

    return {'msg': 'all good'}


@app.get('/recomendation/{user_id}')
def recomendation_5(user_id: str):

    try:
        recomendations = MR.recomendation(user=user_id)

        return {'user_id': user_id,
                'recomendation': recomendations}
    except:
        return {
            "user_id": user_id,
            "recomendation": [
                "43ee88561093499d9e571d4db5f20b79",
                "2265e8aa066cc6c4528d4be900eb5b64",
                "3158c44b08596ff51ee3560fad16cc09",
                "1ae28ef6d0421f92f2e4e6d407e90347",
                "eeba3ee5aa7d1d571752248eb4c81c20"
            ]
        }

@app.get('/NLP/{input}')
def nlp(input:str):
    return {'msg': NLP.getSentimentAnalysis(input=input)}
