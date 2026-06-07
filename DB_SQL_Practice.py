from sqlalchemy import create_engine, text, inspect
import pandas as pd

engine = create_engine('sqlite:///D:/Python_Projects/Django_Projects/expense_tracker/db.sqlite3')
con = engine.connect()
inspector = inspect(engine)
print(inspector.get_table_names())
query = text('SELECT * FROM expense_tracker_app_expensedatamodel')
rs = con.execute(query)
expense_df = pd.DataFrame(rs.fetchall())
print(expense_df.info())
print(expense_df.head())l

# query with Pandas
query_df = pd.read_sql_query('SELECT * FROM expense_tracker_app_expensedatamodel WHERE expense_category LIKE "Shopping"', engine)
print(query_df.head())
con.close()