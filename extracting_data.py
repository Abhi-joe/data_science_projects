import sqlite3

connection= sqlite3.connect("Mydata.db")

cursor= connection.cursor()

query="select * from   StudentData"

cursor.execute(query)

#fetching the data returned by executing the query
result= cursor.fetchall()

#extracting all the data in the resultset pf query
for i in result:
    print(i)

connection.close()