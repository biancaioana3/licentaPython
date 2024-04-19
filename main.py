# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'
#
# if __name__ == '__main__':
#     app.run(debug=True)
import mysql.connector

# Configurația bazei de date
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'HRSystem'
}

try:
    # Conectarea la baza de date
    connection = mysql.connector.connect(**db_config)

    if connection.is_connected():
        print("Conexiunea la baza de date a fost realizată cu succes!")

        # Crearea unui cursor pentru a executa interogările SQL
        cursor = connection.cursor()

        # Executarea unei interogări SELECT pentru a obține toate înregistrările din tabelul 'users'
        cursor.execute("SELECT * FROM job_has_cv")

        # Obținerea descrierii coloanelor
        column_names = [column[0] for column in cursor.description]
        print("Coloanele tabelului 'users':", column_names)

        # Obținerea tuturor rândurilor rezultate și printarea cheilor și valorilor asociate
        users = cursor.fetchall()
        for user in users:
            user_dict = dict(zip(column_names, user))
            print(user_dict)

except mysql.connector.Error as e:
    print("Eroare la conectarea la baza de date:", e)

finally:
    # Închiderea cursorului și a conexiunii
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("Conexiunea la baza de date a fost închisă.")

