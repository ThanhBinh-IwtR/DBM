# from datetime import datetime, timedelta, timezone
# from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
# import pandas as pd
# cho nay tu them account_name, account_key, container_name nha


# connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
# blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# container_client = blob_service_client.get_container_client(container_name)


# def maindata(): 
#     blob_list = []
#     for blob_i in container_client.list_blobs():
#         blob_list.append(blob_i.name)



#     for blob_i in blob_list:
        
#         sas_i = generate_blob_sas(account_name = account_name,
#                                     container_name = container_name,
#                                     blob_name = blob_i,
#                                     account_key=account_key,
#                                     permission=BlobSasPermissions(read=True),
#                                     expiry=datetime.now(timezone.utc) + timedelta(hours=1))
        
#         sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_i + '?' + sas_i
        
#         maindata = pd.read_csv(sas_url)
#         # df_list.append(df)
#         return maindata
    
import pyodbc
import pandas as pd

server = 'databasefordbm.database.windows.net'
database = 'emissions_db'
username = 'name'
password = 'pass'
driver = '{ODBC Driver 18 for SQL Server}'
# Chuỗi kết nối
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

# Kết nối tới Azure SQL Database


def maindata(): 
    conn_azure = pyodbc.connect(conn_str)
    query = 'SELECT * FROM Emissions'
    maindata = pd.read_sql(query, conn_azure)
    conn_azure.close()
    return maindata
# Đọc dữ liệu từ bảng
# query = 'SELECT * FROM Emissions'
# data = pd.read_sql(query, conn_azure)

