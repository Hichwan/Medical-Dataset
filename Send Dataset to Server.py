import pandas as pd
import pyodbc
import traceback

csv_file = "82c67289-d5be-46a2-929a-e05f34fb3cb5.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
except Exception:
    print(f"Failed to load CSV file: {Exception}")
    traceback.print_exc()
    exit()

server = 'testdb-project.c54kgaaiw6cy.us-east-2.rds.amazonaws.com'
database = 'testdb-project'
username = 'admin'
password = 'rich123mond!'
driver = '{ODBC Driver 18 for SQL Server}'

connection_string = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'

try:
    connection = pyodbc.connect(connection_string)
    print("Connection to SQL Server successful!")
except Exception:
    print(f"Failed to cconnect to SQL Server: {Exception}")
    traceback.print_exc()
    exit()

try:
    cursor = connection.cursor()

    table_name = "Hospital_Data_Set_Quarter_2_2024"
    columns = ', '.join(df.columns)

    for index, row in df.iterrows():
        try:
            values =', '.join([f"'{str(value)}'" if not pd.isna(value) else 'NULL' for value in row])
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

            print(f"Executing query: {insert_query}")
       
            cursor.execute(insert_query)

        except Exception:
            print(f"Failed to insert row {index}: {Exception}")
            traceback.print_exc()
    
    connection.commit()
    print("Data successfully inserted into the database!")

except Exception:
    print(f"An error occurred: {Exception}")

finally:
    if 'connection' in locals():
        connection.close()
        print("Database connection closed.")