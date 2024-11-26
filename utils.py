import asyncio
import aiohttp
from gpt4all.gpt4all import sys
import yaml
import json
import argparse

post_sem = asyncio.Semaphore(1)

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                # Attempt to detect encoding, fallback to utf-8
                encoding = response.charset or 'utf-8'
                return await response.text(encoding=encoding, errors='replace')
        except Exception as e:
            print(f"Request failed for {url}: {e}", file=sys.stderr)
            return None

async def post_request(url, data):
    async with post_sem:
        await asyncio.sleep(2)

    print("Sending POST request to " + url)
    async with aiohttp.ClientSession() as session:
        response = await session.post(url=url,
                                      data=json.dumps(data),
                                      headers={"Content-Type": "application/json"})
        return await response.json()

def loadConfig(mode="default"):
    with open("config.yaml", "r") as file:
        configs = yaml.safe_load(file)
    return configs.get(mode)


def loadCliArgs():
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

config = loadConfig()
