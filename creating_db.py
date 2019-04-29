import sqlite3

#creating a connection to db
connection =sqlite3.connect("Mydata.db")

cursor= connection.cursor()

#query to create a table
create_table="""
    create table StudentData(
        sudent_id ineteger primary key,
        student_name varchar(20),
        student_major varchar(10),
        student_grades char(1)
    );
    """
#executing queries
cursor.execute(create_table)

#commiting query
connection.commit()

#closing the connection
connection.close()
