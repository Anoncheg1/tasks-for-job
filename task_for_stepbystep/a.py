""" This program is a task from "Step By Step" copmany for job Machine
Learning Engineer.  This is modified version with fixed errors, added
comments and type hints.

"""

import sqlite3
from sqlite3 import Connection
import sys

def connect_to_db(db_name) -> None | Connection:
    "Create connection to a database file in current directory."
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def create_table(conn: Connection):
    "Create table users."
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER)""")
    conn.commit()

def insert_user(conn: Connection, user_name: str, user_age: int):
    "Insert row to users table."
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)",
                   (user_name, user_age))
    conn.commit()

def creat_and_fill_users(db_name: str):
    "Create table users and insert one row."
    # Usage example
    db_connection = connect_to_db(db_name)
    if db_connection is None:
        sys.exit(1)
    create_table(db_connection)
    insert_user(db_connection, 'Alice', 30)

def test_drop(db_name: str, table_name: str):
    "Drop table if exist."
    db_connection = connect_to_db(db_name)
    cur = db_connection.cursor()
    cur.execute("DROP TABLE IF EXISTS " + table_name)

def test_select(db_name: str, table_name: str):
    "Select and print all rows from table."
    db_connection = connect_to_db(db_name)
    cur = db_connection.cursor()
    cur.execute("SELECT * FROM " + table_name)

    rows = cur.fetchall()
    for row in rows:
        print(row)


if __name__ == "__main__":
    DB_NAME_ARG = 'my_database.db'
    test_drop(DB_NAME_ARG, 'users')
    creat_and_fill_users(DB_NAME_ARG)
    test_select(DB_NAME_ARG, 'users')
