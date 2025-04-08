import csv
import io

def normalize_text(text: str) -> str:
    """
    Normalize text by removing multiple whitespaces and newlines. Only single newline or space is preserved.
    """
    return " ".join(text.split())

class ElementTransformer:
    def transform(self, el) -> str:
        raise NotImplementedError()

class Paragraph(ElementTransformer):
    def transform(self, el) -> str:
        return normalize_text(el.text.strip())

class List(ElementTransformer):
    def transform(self, el) -> str:
        """
        Transforms <ol> and <ul> elements into formatted text.
        Supports nested lists and maintains correct formatting.
        """
        return self._parse_list(el)

    def _parse_list(self, el, level=1, prefix="") -> str:
        items = []
        is_ordered = el.name == "ol"

        for i, li in enumerate(el.find_all("li", recursive=False), start=1):
            # Determine prefix: numbered for <ol>, bullet for <ul>
            item_prefix = f"{prefix}{i}. " if is_ordered else f"{prefix}- "

            # Extract text while preserving indentation
            li_text = li.get_text().strip()

            # Process nested lists, if any
            nested_list = li.find(["ul", "ol"])
            if nested_list:
                nested_text = self._parse_list(nested_list, level + 1, prefix + "   ")
                items.append(f"{item_prefix}{li_text}\n{nested_text}")
            else:
                items.append(f"{item_prefix}{li_text}")

        return "\n".join(items)

class Table(ElementTransformer):
    def transform(self, el) -> str:
        """
        Transform table to csv string
        """
        rows = []
        for row in el.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
            rows.append(cells)

        # Convert to CSV string
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerows(rows)

        csv_string = output.getvalue()
        output.close()        

        return csv_string

