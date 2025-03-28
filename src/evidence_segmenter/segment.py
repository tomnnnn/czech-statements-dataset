from .html_transformers import Paragraph, Table, List
from bs4 import BeautifulSoup

transformation_rules = {
    "p": Paragraph(),
    "table": Table(),
    "ul": List(),
    "ol": List()
}


def transform_element(el) -> dict|None:
    """
    Transform element to string
    """
    tag = el.name
    if tag in transformation_rules:
        return {"tag": tag, "text": transformation_rules[tag].transform(el), "raw": el.prettify()}
    else:
        return None


def segment_article(html: str) -> list:
    """
    Transform html to string
    """
    soup = BeautifulSoup(html, "html.parser")
    return [transform_element(el) for el in soup.find_all()]
