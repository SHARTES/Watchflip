"""Run a SQL file or a query from stdin against the project database.

    python q.py somefile.sql
    python q.py            # then paste SQL, Ctrl-D to run
"""
import os, sys, pathlib, psycopg
from dotenv import load_dotenv

load_dotenv(".env")
sql = pathlib.Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()

with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
    cur = conn.execute(sql)
    if cur.description:
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
        w = [max(len(c), *(len(str(r[i])) for r in rows)) if rows else len(c)
             for i, c in enumerate(cols)]
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        print("  ".join("-" * x for x in w))
        for r in rows:
            print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))
        print(f"\n{len(rows)} row(s)")
    conn.commit()
