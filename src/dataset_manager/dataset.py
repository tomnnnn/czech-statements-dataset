import datetime
import os
import sqlite3

class DemagogDataset:
    """
    Class for managing the dataset of statements and evidence from Demagog.cz
    The dataset is stored in a SQLite database

    The dataset consists of two or more main tables:
    - statements: contains the statements with their metadata
    - evidence: contains the evidence documents for each statement

    Additional tables:
    - tags: contains the tags for each statement
    """
    new = False

    def __init__(self, path, evidence_source="demagog", readonly=False):
        """
        Initialize the dataset database connection, create tables if they don't exist
        """
        self.path = path

        if not os.path.exists(path):
            self.new = True

        self.conn = sqlite3.connect(f"file:{path}?mode={'ro' if readonly else 'wa'}", uri=True)
        self.readonly = readonly
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.set_evidence_source(evidence_source)

        if not readonly:
            # create tables if not exist
            self.cursor.execute(
                """CREATE TABLE IF NOT EXISTS statements (
                id INTEGER PRIMARY KEY,
                statement TEXT,
                label TEXT,
                author TEXT,
                date TEXT,
                party TEXT,
                explanation TEXT,
                explanation_brief TEXT,
                origin TEXT
                )""",
            )

            # NOTE: For the sake of simplicity, no join table is used, despite many-to-many relationship
            self.cursor.execute(
                """CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                statement_id INTEGER,
                tag TEXT
                )
                """
            )
            # create index
            self.cursor.execute(
                """CREATE INDEX IF NOT EXISTS idx_statements_id ON statements (id)"""
            )

            self.cursor.execute(
                """CREATE INDEX IF NOT EXISTS idx_tags_statement_id ON tags (statement_id)"""
            )

            self.conn.commit()

    def set_evidence_source(self, source):
        """
        Set the source of the evidence documents. Creates a new evidence table for the source if it doesn't exist
        """
        self.evidence_source = source
        self._evidence_table = "evidence_" + source

        if not self.readonly:
            # create evidence table if not exist
            self.cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._evidence_table} (
                id INTEGER PRIMARY KEY,
                statement_id INTEGER,
                url TEXT,
                title TEXT,
                description TEXT,
                content TEXT,
                author TEXT,
                type TEXT,
                published TEXT,
                source TEXT,
                accessed TEXT,
                FOREIGN KEY (statement_id) REFERENCES statements (id)
                )"""
            )

            # create index
            self.cursor.execute(
                f"""CREATE INDEX IF NOT EXISTS idx_{self._evidence_table}_statement_id ON {self._evidence_table} (statement_id)"""
            )

            self.conn.commit()


    def get_all_statements(self, allowed_labels=None, min_evidence_count=0):
        """
        Get all statements from the dataset

        Args:
        allowed_labels: Optional list of labels to filter the statements by
        min_evidence_count: Optional minimum number of evidence documents required for the statement

        """
        allowed_labels_str = " WHERE LOWER(s.label) IN ({})".format( ", ".join([f"'{l}'" for l in allowed_labels])) if allowed_labels else ""
        print(allowed_labels_str)

        self.cursor.execute(f"""
            SELECT s.* 
            FROM statements s
            JOIN {self._evidence_table} a ON s.id = a.statement_id
            {allowed_labels_str}
            GROUP BY s.id
            HAVING COUNT(a.id) >= {min_evidence_count}
        """)

        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def get_statement(self, statement_id):
        """
        Get a statement by its ID
        """
        self.cursor.execute("SELECT * FROM demagog WHERE id = ?", (statement_id,))

        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_evidence(self, statement_id):
        """
        Get all evidence documents for a statement by the statement ID
        """
        self.cursor.execute(
            f"SELECT * FROM {self._evidence_table} WHERE statement_id = {str(statement_id)}"
        )

        row = self.cursor.fetchall()
        return [dict(r) for r in row]

    def get_all_evidence(self):
        """
        Get all evidence documents from the dataset
        """
        self.cursor.execute("SELECT * FROM ?", (self._evidence_table))

        rows = self.cursor.fetchall()
        return [dict(r) for r in rows]

    def insert_evidence(self, statement_id, evidence: dict):
        """
        Insert an evidence document into the dataset
        """
        self.cursor.execute(
            f"""INSERT INTO {self._evidence_table} 
            VALUES (statement_id, url, title, description, content, author, type, published, source, accessed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                statement_id,
                evidence.get("url", ""),
                evidence.get("title", ""),
                evidence.get("description", ""),
                evidence.get("content", ""),
                evidence.get("author", ""),
                evidence.get("type", ""),
                evidence.get("published", ""),
                evidence.get("source", ""),
                datetime.datetime.now(),
            ),
        )
        self.conn.commit()

    def insert_statement(self, statement: dict):
        """
        Insert a statement into the dataset
        """
        self.cursor.execute(
            """ INSERT INTO demagog 
            VALUES (statement, label, author, date, party, explanation, explanation_brief, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                statement.get("statement", ""),
                statement.get("label", ""),
                statement.get("author", ""),
                statement.get("date", ""),
                statement.get("party", ""),
                statement.get("explanation", ""),
                statement.get("explanation_brief", ""),
                statement.get("origin", ""),
            ),
        )

        # save tags
        self.cursor.executemany(
            f"""INSERT INTO tags (statement_id, tag) VALUES (?, ?)""",
            [
                (statement["id"], tag)
                for tag in statement.get("tags", [])
            ]
        )
        self.conn.commit()


    def insert_statements(self, statements: list):
        """
        Insert multiple statements into the dataset
        """
        self.cursor.executemany(
            """ INSERT INTO statements 
            (statement, label, author, date, party, explanation, explanation_brief, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    statement.get("statement", ""),
                    statement.get("label", ""),
                    statement.get("author", ""),
                    statement.get("date", ""),
                    statement.get("party", ""),
                    statement.get("explanation", ""),
                    statement.get("explanation_brief", ""),
                    statement.get("origin", ""),
                )
                for statement in statements
            ],
        )

        # save tags
        self.cursor.executemany(
            f"""INSERT INTO tags (statement_id, tag) VALUES (?, ?)""",
            [
                (statement["id"], tag)
                for statement in statements
                for tag in statement.get("tags", [])
            ]
        )

        self.conn.commit()

    def insert_evidence_batch(self, statement_id, evidence: list):
        """
        Insert multiple evidence documents into the dataset
        """
        self.cursor.executemany(
            f"""INSERT INTO {self._evidence_table} 
            (statement_id, url, title, description, content, author, type, published, source, accessed) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    statement_id,
                    ev.get("url", ""),
                    ev.get("title", ""),
                    ev.get("description", ""),
                    ev.get("content", ""),
                    ev.get("author", ""),
                    ev.get("type", ""),
                    ev.get("published", ""),
                    ev.get("source", ""),
                    datetime.datetime.now(),
                )
                for ev in evidence
            ],
        )

        self.conn.commit()
    def get_scraped_statement_ids(self):
        """
        Get a list of statement_ids that have already been scraped (have atleast one evidence document)
        """
        self.cursor.execute(f"SELECT DISTINCT statement_id FROM {self._evidence_table}")
        result = self.cursor.fetchall()

        # Return a list of unique statement_ids
        return [r[0] for r in result]


    def __del__(self):
        self.conn.close()
