import asyncio
import bs4
import json
from tqdm.asyncio import tqdm
from collections import defaultdict
import sys
from utils import fetch

class ArticleRetriever:
    def __init__(self, fetch_concurrency=5, fetch_delay=0.5):
        self.fetch_sem = asyncio.Semaphore(fetch_concurrency)
        self.fetch_delay = fetch_delay

    def __extract_article(self, html, max_num_paragraphs=1000, min_num_words=5):
        """
        Extract article content from HTML using a combination of article tags and paragraph heuristics

        Args:
            html (str): The HTML content to extract the article from.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.
        """

        try:
            soup = bs4.BeautifulSoup(html, "html.parser", parse_only=bs4.SoupStrainer(["title","body"]))
        except Exception as e:
            raise e

        parsed_article = {
            "title": "",
            "content": ""
        }

        if soup.find("article") is not None:
            article = soup.find("article")
            article_text = article.get_text(strip=True)
            title = soup.find('title')
            title = title.get_text() if title else ""

            parsed_article["title"] = title
            parsed_article["content"] = article_text
            return parsed_article

        # no article found, resort to paragraph heuristics
        p_parents = defaultdict(list)

        ps = soup.find_all("p")[:max_num_paragraphs]

        for p in ps:
            p_parents[p.parent].append(p)

        parents_counts = sorted([(parent, len(ps)) for parent, ps in p_parents.items()], key=lambda v: -v[1])
        if not parents_counts:
            raise ValueError("No paragraphs found")

        article_dom = parents_counts[0][0]
        article_text = " ".join(p.get_text().strip() for p in article_dom.find_all("p") if len(p.get_text().split()) > min_num_words)
        title = soup.find('title')
        title = title.get_text() if title else ""

        parsed_article["title"] = title
        parsed_article["content"] = article_text

        return parsed_article


    async def batch_retrieve(self, links: list[str], id=None, max_num_paragraphs: int=1000, min_num_words:int=5, show_progress=True):
        """
        Extract article content from a batch of URLs.

        Args:
            links (List[Dict[str, str]]): A list of dictionaries containing the URL and ID of the article.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.

        Returns:
            dict|list: If an ID is provided, a dictionary containing the ID and a list of articles. Otherwise, a list of articles.
        """
        tasks = [self.retrieve(link, max_num_paragraphs, min_num_words) for link in links]
        articles = await tqdm.gather(*tasks, desc="Retrieving and extracting articles", unit="article", disable=not show_progress)
        articles = [r for r in articles if r is not None]

        return {"id": id, "articles": articles} if id else articles


    async def batch_retrieve_save(self, path:str, links: list[str], id=None, max_num_paragraphs: int=1000, min_num_words:int=5, show_progress=True):
        """
        Extract article content from a batch of URLs and save it to a JSON file.

        Args:
            path (str): The path to save the JSON file.
            links (List[Dict[str, str]]): A list of dictionaries containing the URL and ID of the article.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.

        Returns:
            None
        """
        articles = await self.batch_retrieve(links, id, max_num_paragraphs, min_num_words, show_progress)
        with open(path, "w") as f:
            json.dump(articles, f, indent=4, ensure_ascii=False)

        return None


    async def retrieve(self, url, max_num_paragraphs=1000, min_num_words=5):
        """
        Fetch and extract article content from a URL

        Args:
            url (str): The URL to extract content from.
            max_num_paragraphs (int): The maximum number of paragraphs to consider.
            min_num_words (int): The minimum number of words in a paragraph to consider.
            id (str): The ID of the article.
        """
        async with self.fetch_sem:
            html = await fetch(url)
            if not html:
                print('Failed to fetch', url, file=sys.stderr)
                return None
            await asyncio.sleep(self.fetch_delay)

        try:
            article = self.__extract_article(html, max_num_paragraphs, min_num_words)
        except Exception as e:
            print(f"Failed to extract article from {url}: {e}", file=sys.stderr)
            return None

        result = {
            "url": url,
            **article,
        }

        return result
