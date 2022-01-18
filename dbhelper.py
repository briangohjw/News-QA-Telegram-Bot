import sqlite3


class DBHelper:
    def __init__(self, dbname="user_qa.sqlite"):
        self.dbname = dbname

    def setup(self):
        conn = sqlite3.connect(self.dbname)
        stmt = '''CREATE TABLE IF NOT EXISTS user_story
            (
            id INTEGER NOT NULL PRIMARY KEY UNIQUE,
            story TEXT
            );
            '''
        conn.execute(stmt)
        stmt = '''CREATE TABLE IF NOT EXISTS user_question
            (
            id INTEGER NOT NULL PRIMARY KEY UNIQUE,
            question TEXT
            );
            '''
        conn.execute(stmt)
        conn.commit()

    def add_story(self, user_id, story):
        conn = sqlite3.connect(self.dbname)
        user_list = conn.execute('SELECT id FROM user_story WHERE id = ?', (user_id,)).fetchall()
        if len(user_list) == 0:
            conn.execute('INSERT INTO user_story (id, story) VALUES (?,?)', (user_id,story))
        else:
            conn.execute('UPDATE user_story SET story = ? WHERE id = ?', (story, user_id))
        conn.commit()

    def get_story(self, user_id):
        conn = sqlite3.connect(self.dbname)
        stmt = "SELECT story FROM user_story WHERE id = ?"
        args = (user_id, )
        story = conn.execute(stmt, args).fetchall()
        if story:
            return story[0]
        else:
            return None

    def add_question(self, user_id, question):
        conn = sqlite3.connect(self.dbname)
        user_list = conn.execute('SELECT id FROM user_question WHERE id = ?', (user_id,)).fetchall()
        if len(user_list) == 0:
            conn.execute('INSERT INTO user_question (id, question) VALUES (?,?)', (user_id,question))
        else:
            conn.execute('UPDATE user_question SET question = ? WHERE id = ?', (question, user_id))
        conn.commit()

    def get_question(self, user_id):
        conn = sqlite3.connect(self.dbname)
        stmt = "SELECT question FROM user_question WHERE id = ?"
        args = (user_id, )
        question = conn.execute(stmt, args).fetchall()
        if question:
            return question[0]
        else:
            return None

