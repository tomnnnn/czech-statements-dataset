from dataset_manager.orm import *
from sqlalchemy import insert
import sqlite3

dataset = init_db("datasets/dataset_new.sqlite")
old_db = sqlite3.connect("datasets/dataset.sqlite")
old_db.row_factory = sqlite3.Row
cur = old_db.cursor()

# --- INSERT STATEMENTS ---

statements = cur.execute("SELECT * FROM statements")
statements_dict = [dict(statement) for statement in statements]

insert_stmt = insert(Statement).values(statements_dict)
dataset.execute(insert_stmt)
dataset.commit()

# --- INSERT ARTICLES ---

articles = cur.execute("SELECT * FROM evidence_demagog")
articles_dicts = [dict(article) for article in articles]

relevances = []
for article_dict in articles_dicts:
    relevance = {
        "article_id": article_dict["id"],
        "statement_id": article_dict.pop("statement_id")
    }


insert_article = insert(Article).values(articles_dicts)
insert_relevance = insert(ArticleRelevance).values(relevances)
dataset.execute(insert_article)
dataset.execute(insert_relevance)

dataset.commit()
