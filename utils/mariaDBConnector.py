# Module Imports
import json
import mariadb


class MariaDBConnector:
    def __init__(self, db_path):
        with open(db_path, 'r', encoding='utf-8') as file:
            db_data = json.load(file)
        self.config = {
            'user': db_data.get("Id"),
            'password': db_data.get("Password"),
            'host': db_data.get("Ip"),
            'port': db_data.get("Port"),
            'database': db_data.get("Database")
        }
        self.table_name = db_data.get("Table")
        self.conn = None
        self.cur = None

    def connection(self):
        self.conn = mariadb.connect(**self.config)
        self.cur = self.conn.cursor()

    def disconnection(self):
        self.cur.close()
        self.conn.close()

    def select_list(self, query):
        try:
            self.cur.execute(query)
            column_names = self.cur.description
        except mariadb.Error as err:
            print(f"오류: {err}")
        return self.cur.fetchall()

    def update_list(self, query):
        try:
            self.cur.execute(query)
            # 커밋
            self.conn.commit()
        except mariadb.Error as err:
            print(f"오류: {err}")



