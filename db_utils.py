import sqlite3
import lclib

def create_db():
    df = lclib.load_training_data(full_dataset=True)
    del df['desc']
    del df['url']
    conn = sqlite3.connect('lcdata.db')
    c = conn.curser()

    cols = ','.join(df.columns)
    sql = '''CREATE TABLE lcdata ({columns})'''.format(columns=cols)
    c.execute(sql)
