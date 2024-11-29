import aiohttp
import sys
import yaml
import json
import argparse
import asyncio
import itertools
from googlesearch import search


async def fetch(url, retries=0):
    timeout=aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            headers = {'User-Agent': config['UserAgent']}
            async with session.get(url, headers=headers) as response:
                if(response.status == 429 and retries < 3):
                    # retry after delay
                    await asyncio.sleep(3, retries+1)
                    return await fetch(url)

                response.raise_for_status()
                encoding = response.charset or 'utf-8'
                return await response.text(encoding=encoding, errors='replace')
        except Exception as e:
            print(f"Request failed for {url}: {e}", file=sys.stderr)
            return None

async def post_request(url, data):
    async with aiohttp.ClientSession() as session:
        response = await session.post(url=url,
                                      data=json.dumps(data),
                                      headers={"Content-Type": "application/json"})
        return await response.json()

def load_config(mode="default"):
    with open("config.yaml", "r") as file:
        configs = yaml.safe_load(file)
    return configs.get(mode)


def load_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Output file name", default="statements.json"
    )
    parser.add_argument(
        "--config-preset",
        help="Configuration preset to use",
        default="default"
    )
    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        default=4, type=int
    )
    return parser.parse_args()


async def track_progress(coro, counter, total, unit=""):
    result = await coro
    print(f"{next(counter)}/{total} {unit}s completed", end="\r", flush=True)

    return result


async def search_google(query, num_results):
    print("Searching Google for:", query)
    return await asyncio.get_event_loop().run_in_executor(None, search, query, num_results)


async def search_bing(query, api_key, num_results):
    params = {
        "q": query,
        "count": num_results,
    }
    endpoint = "https://api.bing.microsoft.com/v7.0/search?"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, headers=headers, params=params) as response:
            if response.status != 200:
                print(f"Failed to fetch Bing search results: {response.status}", file=sys.stderr)
                return []
            else:
                result = await response.json()
                if 'webPages' not in result:
                    return []
                return result['webPages']['value']

config = load_config()
