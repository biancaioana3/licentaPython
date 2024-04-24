import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('mysql+mysqlconnector://root:@localhost/HRSystem')

csv_file = 'Resume_data.csv'
table_name = 'data_model'

data = pd.read_csv(csv_file)

conn = engine.connect()

max_id = 0

if max_id is None:
    max_id = 0

max_id += 1

data['id'] = range(max_id, max_id + len(data))

data.to_sql(table_name, con=engine, if_exists='replace', index=False)

conn.close()

print("Datele au fost inserate cu succes în baza de date.")
