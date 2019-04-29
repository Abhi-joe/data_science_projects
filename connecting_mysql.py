import pymysql

config={
    'host': '127.0.0.1',
    'username':'root',
    'password': 'abhi',
    'db':'ecommdb'
}

connection= pymysql.connect(config['host'],config['username'], config['password'], config['db'])

cursor=connection.cursor()
query='select * from address'
cursor.execute(query)
result=cursor.fetchall()

for i in result:
    print(i)

connection.close()