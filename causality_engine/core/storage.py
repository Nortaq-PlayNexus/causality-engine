"""Storage."""
import sqlite3, json, os
from .causal_graph import CausalGraph

class Storage:
    def __init__(self, db_path):
        self.db_path=db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        con=sqlite3.connect(db_path)
        con.execute("CREATE TABLE IF NOT EXISTS graphs(id INTEGER PRIMARY KEY,name TEXT,data TEXT)")
        con.commit(); con.close()
    def save(self,name,graph):
        data=json.dumps(graph.to_dict())
        con=sqlite3.connect(self.db_path)
        cur=con.execute("SELECT id FROM graphs WHERE name=?",(name,))
        if cur.fetchone():
            con.execute("UPDATE graphs SET data=? WHERE name=?",(data,name))
        else:
            con.execute("INSERT INTO graphs(name,data) VALUES(?,?)",(name,data))
        con.commit(); con.close()
    def load(self,name):
        con=sqlite3.connect(self.db_path)
        row=con.execute("SELECT data FROM graphs WHERE name=?",(name,)).fetchone()
        con.close()
        if not row: return None
        return CausalGraph.from_dict(json.loads(row[0]))
    def list_graphs(self):
        con=sqlite3.connect(self.db_path)
        rows=con.execute("SELECT name FROM graphs").fetchall()
        con.close()
        return [r[0] for r in rows]
