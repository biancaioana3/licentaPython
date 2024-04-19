from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector

app = Flask(__name__)
CORS(app)  # Permit toate originile

# Configurațiile bazei de date MySQL
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'HRSystem'
}


@app.route('/recomanda-candidati', methods=['POST'])
def recomanda_candidati():
    global user_dict
    try:
        # Conectarea la baza de date
        connection = mysql.connector.connect(**db_config)

        if connection.is_connected():
            print("Conexiunea la baza de date a fost realizată cu succes!")

            # Crearea unui cursor pentru a executa interogările SQL
            cursor = connection.cursor()

            # Executarea unei interogări SELECT pentru a obține toate înregistrările din tabelul 'users'
            cursor.execute("SELECT * FROM users")

            # Obținerea tuturor rândurilor rezultate și printarea lor
            # users = cursor.fetchall()
            # for user in users:
            #     print(user)

            # Obținerea descrierii coloanelor
            column_names = [column[0] for column in cursor.description]
            # print("Coloanele tabelului 'users':", column_names)

            # Obținerea tuturor rândurilor rezultate și printarea cheilor și valorilor asociate
            users_dicts = []
            users = cursor.fetchall()
            for user in users:
                user_dict = dict(zip(column_names, user))
                users_dicts.append(user_dict)

            user_data = [{'name': user[0], 'email': user[1]} for user in users]
            return jsonify(users_dicts)

    except mysql.connector.Error as e:
        return jsonify({"Eroare la conectarea la baza de date:", e})


    finally:
        # Închiderea cursorului și a conexiunii
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Conexiunea la baza de date a fost închisă.")

    # connection = mysql.connector.connect(**db_config)
    # cursor = connection.cursor()
    #
    # # Interogare pentru a obține doar coloanele name și email din tabela users
    # cursor.execute("SELECT * FROM users")
    # users = cursor.fetchall()


    # # Închide conexiunea și cursorul
    # cursor.close()
    # connection.close()
    #
    # # Transformă rezultatul în format JSON și returnează
    # user_data = [{'name': user[0], 'email': user[1]} for user in users]
    # # return jsonify({'users': user_data})
    # # Primește datele CV-ului din request
    # cv_data = request.json
    #
    # # Aici poți adăuga logica ta de machine learning pentru a genera recomandări
    # # În acest exemplu, vom returna un JSON cu o listă de recomandări
    # recomandari = ["Candidatul A", "Candidatul B", "Candidatul C"]
    # # Returnează recomandările sub formă de JSON
    # # return jsonify({'recomandari': recomandari})
    # return jsonify({'users': user_data})

if __name__ == '__main__':
    app.run(debug=True)
