import aiohttp
import sys
import envyaml
import json
import argparse
import asyncio
from googlesearch import search

async def fetch(url, retries=0):
    timeout=aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            headers = {'User-Agent': config['UserAgent']}
            async with session.get(url, headers=headers) as response:
                if(response.status == 429 and retries < 3):
                    # retry after cooldown
                    print("Retrying")
                    await asyncio.sleep(3)
                    return await fetch(url, retries+1)

                response.raise_for_status()
                encoding = response.charset or 'utf-8'

                return await response.text(encoding=encoding, errors='replace')
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
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.post(url=url,
                                          data=json.dumps(data),
                                          headers={"Content-Type": "application/json"})
        return await response.json()
    except TimeoutError as e:
        print(f"Request timed out for {url}: {e}", file=sys.stderr)
        if retries < 3:
            await asyncio.sleep(5)
            return await post_request(url, data, retries+1)
        return None

def load_config(mode="default"):
    configs = envyaml.EnvYAML('config.yaml', strict=False)
    return configs[mode]


def load_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preset", help="Config preset name", default="default"
    )
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose output", action="store_true"
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
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(endpoint, headers=headers, params=params, timeout=timeout) as response:
            if response.status != 200:
                print(f"Failed to fetch Bing search results: {response.status}", file=sys.stderr)
                return []
            else:
                try:
                    result = await response.json()
                    if 'webPages' not in result:
                        return []

                    return result['webPages']['value']
                except Exception as e:
                    print(f"Failed to fetch Bing search results: {e}", file=sys.stderr)
                    return []

def verbose_print(*args, **kwargs):
    if config['Verbose']:
        print(*args, **kwargs, file=sys.stderr)


args = load_cli_args()
print("Loading config preset:", args.preset)
config = load_config(args.preset)
config['Verbose'] = args.verbose
