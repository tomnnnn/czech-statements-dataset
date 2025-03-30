import datetime
import os
import sqlite3
from bs4 import BeautifulSoup

class Dataset:
    def set_evidence_source(self, source):
        pass

    def get_statements(self, allowed_labels=None, min_evidence_count=0):
        pass

    def get_statement(self, statement_id):
        pass

    def get_articles(self, statement_id = None):
        pass

    def get_article(self, article_id):
        pass

    def get_segments(self, article_id=None):
        pass

    def get_segment(self, segment_id):
        pass

    def get_segment_relevances(self):
        pass

    def get_article_relevances(self):
        pass

    def set_segment_relevance(self, segment_id, statement_id, relevance=1):
        pass

    def set_segment_relevances(self, segment_relevances):
        pass

    def set_article_relevance(self, article_id, statement_id, relevance=1):
        pass

    def set_article_relevances(self, article_relevances):
        pass

    def delete_statement(self, statement_id):
        pass

    def delete_article(self, article_id):
        pass

    def delete_segment(self, segment_id):
        pass

    def insert_statement(self, statement):
        pass

    def insert_statements(self, statements):
        pass

    def insert_article(self, article):
        pass

    def insert_articles(self, articles):
        pass

    def insert_segment(self, segment):
        pass

    def insert_segments(self, segments):
        pass


class DemagogDataset(Dataset):
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

        self.conn = sqlite3.connect(f"file:{path}?mode={'ro' if readonly else 'rwc'}", uri=True)
        self.readonly = readonly
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.set_evidence_source(evidence_source)

    def set_evidence_source(self, source):
        """
        Set the source of the evidence documents. Creates a new evidence table for the source if it doesn't exist
        """
        self.evidence_source = source
        self._evidence_table = "articles_" + source
        self._segment_table = "segments_" + source


    def get_all_statements(self, allowed_labels=None, min_evidence_count=0):
        """
        Get all statements from the dataset

        Args:
        allowed_labels: Optional list of labels to filter the statements by
        min_evidence_count: Optional minimum number of evidence documents required for the statement

        """
        allowed_labels_str = " WHERE LOWER(s.label) IN ({})".format( ", ".join([f"'{l}'" for l in allowed_labels])) if allowed_labels else ""

        # TODO: tidy up
        if min_evidence_count == 0:
            self.cursor.execute(f"""
                SELECT * 
                FROM statements s
                {allowed_labels_str}
            """)
        else:
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
        self.cursor.execute("SELECT * FROM statements WHERE id = ?", (statement_id,))

        row = self.cursor.fetchone()
        return dict(row) if row else None

    def _convert_html_to_text(self, html):
        """
        Convert HTML content to plain text
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Decompose all <img> and <figure> tags in a single loop
        for tag in soup.find_all(['img', 'figure']):
            tag.decompose()

        # Remove <p> tags containing 'cookie' in a single pass
        for tag in soup.find_all('p'):
            if 'cookie' in tag.text.lower():
                tag.decompose()

        # Collect text efficiently using a generator
        text = '\n'.join(
            tag.text.strip()
            for tag in soup.find_all('p')
            if len(tag.text.strip()) > 100
        )

        return text

    def get_evidence(self, statement_id):
        """
        Get all evidence articles for a statement by the statement ID
        """
        self.cursor.execute(
            f"SELECT * FROM {self._evidence_table} WHERE statement_id = {str(statement_id)}"
        )

        row = self.cursor.fetchall()

        result = [dict(r) for r in row]

        # NOTE: temporarily convert html content to plain text in demagog dataset
        if self.evidence_source == "demagog":
            result = [
                {**r, "content": self._convert_html_to_text(r["content"])}
                for r in result
            ]

        return result

    def get_evidence_by_id(self, evidence_id):
        self.cursor.execute(f"SELECT * FROM {self._evidence_table} WHERE id = ?", (evidence_id,))

        return dict(self.cursor.fetchone())


    def get_all_evidence(self):
        """
        Get all evidence documents from the dataset
        """
        self.cursor.execute(f"SELECT * FROM {self._evidence_table}")

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


    def delete_statement(self, statement_id):
        """
        Delete a statement and its evidence documents from the dataset
        """
        self.cursor.execute(f"DELETE FROM statements WHERE id = ?", (statement_id,))
        self.cursor.execute(f"DELETE FROM {self._evidence_table} WHERE statement_id = ?", (statement_id,))
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

    def get_segment_relevances(self, with_statement=False, with_segment=False):
        """
        Get segment - statement pair relevances.

        :param with_statement: If True, join with the statement table to get statement details.
        :param with_segment: If True, join with the segment table to get segment details.
        :return: List of dictionaries representing segment relevances.
        """
        query = """
            SELECT sr.*{statement_col}{segment_col}
            FROM segment_relevance sr
            {statement_join}
            {segment_join}
        """.format(
            statement_col=", s.statement AS statement" if with_statement else "",
            segment_col=", seg.text AS text" if with_segment else "",
            statement_join="JOIN statements s ON sr.statement_id = s.id" if with_statement else "",
            segment_join=f"JOIN {self._segment_table} seg ON sr.segment_id = seg.id" if with_segment else ""
        )

        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]


    def set_segment_relevance(self, segment_id, statement_id, relevance=1):
        """
        Sets or updates the relevance of a segment to a statement
        """

        self.cursor.execute(
            "INSERT OR REPLACE INTO segment_relevance (segment_id, statement_id, relevance) VALUES (?, ?, ?)",
            (segment_id, statement_id, relevance)
        )
        self.conn.commit()
        print(f"Segment {segment_id} successfully attached to statement {statement_id} with relevance {relevance}")

    def set_segment_relevance_batch(self, segment_relevances):
        """
        Sets or updates the relevance of multiple segments to statements
        """

        self.cursor.executemany(
            "INSERT OR REPLACE INTO segment_relevance (segment_id, statement_id, relevance) VALUES (?, ?, ?)",
            segment_relevances
        )
        self.conn.commit()
        print(f"Segment relevances successfully updated")


    def __del__(self):
        self.conn.close()
