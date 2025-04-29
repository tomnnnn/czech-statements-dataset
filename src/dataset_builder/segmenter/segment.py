from .html_transformers import Paragraph, Table, List
from bs4 import BeautifulSoup, Tag

transformation_rules = {
    "p": Paragraph(),
    "table": Table(),
    "ul": List(),
    "ol": List()
}


def transform_element(el: Tag) -> str:
    """
    Transform HTML element to string segment
    """
    tag = el.name
    if tag in transformation_rules:
        return transformation_rules[tag].transform(el)
    else:
        return ""

def extract_elements(html: str, title: str) -> list[tuple[str, Tag]]:
    """
    Extract elements from HTML and associate them with the nearest heading.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = []

    current_heading = title

    for el in soup.find_all():
        if isinstance(el, Tag):
            if el.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                current_heading = el.get_text(strip=True)
            else:
                result.append((current_heading, el))

    return result

def segment_article(html: str, title: str, min_len = 25,) -> list[str]:
    """
    Transform html to list of string segments
    """

    elements = extract_elements(html, title=title)

    segments = [transform_element(el[1]) for el in elements]
    segments = [seg for seg in segments if len(seg) > min_len]

    return segments
