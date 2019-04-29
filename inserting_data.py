import sqlite3

connection= sqlite3.connect("Mydata.db")

cursor=connection.cursor()

student_data=[
    (1,"Alex", "Computer Sc", "A"),
    (2,"Adam", "Electrical", "A"),
    (3,"Elizabeth", "Operations", "A"),
    (4,"Rebecca", "Computer Sc", "A"),
    (5,"Teressa", "Computer Sc", "A")
]


#code for debugging
for stud in student_data:
    print(stud)

for student in student_data:
    insert_data="""
        insert into StudentData values({0}, "{1}", "{2}", "{3}");
    """.format(student[0], student[1], student[2], student[3])

    cursor.execute(insert_data)

connection.commit()

connection.close()