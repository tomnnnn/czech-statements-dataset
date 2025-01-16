import aiohttp
import sys
import argparse
import asyncio
from utils.utils import *

async def fetch(url, retries=0, ):
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.3"}
            async with session.get(url, headers=headers) as response:
                if response.status == 429 and retries < 3:
                    # retry after cooldown
                    print("Retrying")
                    await asyncio.sleep(3)
                    return await fetch(url, retries + 1)

                response.raise_for_status()
                encoding = response.charset or "utf-8"

                return await response.text(encoding=encoding, errors="replace")
        except asyncio.exceptions.TimeoutError as e:
            print(f"Request timed out for {url}: {e}", file=sys.stderr)
            return None
        except aiohttp.ClientError as e:
            print(f"Request failed for {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Request failed for {url}: {type(e)}", file=sys.stderr)
            return None


async def post_request(url, data, retries=0):
    print("Posting request to", url, file=sys.stderr)

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        status=response.status,
                        message=f"Error {response.status}: {response.reason}",
                        request_info=response.request_info,
                        history=response.history,
                    )
                articles = await response.json()  # Parse JSON response

                print("Received", file=sys.stderr)
                return articles
        except asyncio.exceptions.TimeoutError as e:
            print(f"Request timed out for {url}: {e}", file=sys.stderr)

            if retries < 3:
                # exponential backoff
                await asyncio.sleep(2 ** retries)
                return await post_request(url, data, retries + 1)
            return None
        except aiohttp.ClientError as exc:
            print(f"An error occurred: {exc}", file=sys.stderr)
            raise


async def search_bing(query, api_key, num_results):
    params = {
        "q": query,
        "count": num_results,
    }
    endpoint = "https://api.bing.microsoft.com/v7.0/search?"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    async with aiohttp.ClientSession() as session:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(
            endpoint, headers=headers, params=params, timeout=timeout
        ) as response:
            if response.status != 200:
                print(
                    f"Failed to fetch Bing search results: {response.status}",
                    file=sys.stderr,
                )
                return []
            else:
                try:
                    result = await response.json()
                    if "webPages" not in result:
                        return []

                    return result["webPages"]["value"]
                except Exception as e:
                    print(f"Failed to fetch Bing search results: {e}", file=sys.stderr)
                    return []


def load_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--preset", help="Config preset name", default="default")
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
    )
    return parser.parse_args()
