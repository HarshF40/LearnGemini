##creating a database

import sqlite3

##connect to sqlite
connection = sqlite3.connect("students.db")

##create a cursor object which is responsible to insert record,create table,retrieve
cursor = connection.cursor()

##create a table
table_info="""
Create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

##Insert some records
cursor.execute('''Insert Into STUDENT values('Harsh','AI','A',100)''')
cursor.execute('''Insert Into STUDENT values('Johny','Web Development','A',90)''')
cursor.execute('''Insert Into STUDENT values('Ankur','App Development','B',100)''')
cursor.execute('''Insert Into STUDENT values('Gaurang','Full Stack Development','B',95)''')
cursor.execute('''Insert Into STUDENT values('Jared','Front End Development','C',100)''')

##to display all the records (not necessary)
print("The inserted records are: ")

data = cursor.execute('''Select * From STUDENT''')

for row in data:
    print(row)

connection.commit()
connection.close()