import { readFile } from "fs/promises";
import { extract } from "@extractus/article-extractor";
import { setTimeout } from "timers/promises";
import pLimit from "p-limit";
import { URL } from "url";
import cliProgress from "cli-progress";
import UserAgent from "user-agents";
import {HeaderGenerator} from 'header-generator';
import fs from 'fs';

let headerGenerator = new HeaderGenerator({
        browsers: [
            {name: "firefox", minVersion: 80},
            {name: "chrome", minVersion: 87},
            "safari"
        ],
        devices: [
            "desktop"
        ],
        operatingSystems: [
            "windows"
        ]
});

// Limit settings
const REQUESTS_PER_DOMAIN_PER_SECOND = 2;
const DOMAIN_LIMITERS = new Map();

// Create progress bar
const progressBar = new cliProgress.SingleBar(
  {
    format:
      "Processing [{bar}] {percentage}% | {value}/{total} URLs | ETA: {eta}s",
    barCompleteChar: "=",
    barIncompleteChar: " ",
    hideCursor: false,
    stream: process.stderr,
  },
  cliProgress.Presets.shades_classic,
);

// Function to create a limiter for each domain
function getLimiterForDomain(domain) {
  if (!DOMAIN_LIMITERS.has(domain)) {
    DOMAIN_LIMITERS.set(domain, pLimit(REQUESTS_PER_DOMAIN_PER_SECOND));
  }
  return DOMAIN_LIMITERS.get(domain);
}

// Function to check if a URL is valid
function isValidUrl(url) {
  try {
    new URL(url);
    return true;
  } catch (error) {
    return false;
  }
}

// Processing each URL with rate limiting
async function processUrl(url) {
  if (!isValidUrl(url)) {
    console.warn(`Skipping invalid URL: ${url}`);
    return;
  }

  const domain = new URL(url).hostname;
  const limiter = getLimiterForDomain(domain);

  return limiter(async () => {
    await setTimeout(1000 / REQUESTS_PER_DOMAIN_PER_SECOND); // Rate limit delay

    let headers = headerGenerator.getHeaders({
        operatingSystems: [
            "linux"
        ],
        locales: ["en-US", "en"]
    });

    headers.proxy = "http://pcEE0ReXWA-res-de:PC_7UoLO7QZqfrAD6rlW@proxy-eu.proxy-cheap.com:5959"

    try {
      const article = await extract(url,{
        signal: AbortSignal.timeout(5000)
      },{
        headers: headers, 
      });

      console.error(`Extracted article from ${url}`);

      return {
        success: true,
        data: article,
      }
    } catch (error) {
      return {
        success: false,
        data: {
          url: url,
          error: error.message,
        }
      }
    }
    finally {
        progressBar.increment(); // Update progress bar
    }

  });
}

async function extractArticles() {
  // Get the file path from the argument
  const args = process.argv.slice(2);
  if (args.length !== 1) {
    console.error('Please provide the path to the JSON file.');
    process.exit(1);
  }

  const filePath = args[0];

  // Read and parse the JSON file
  let urls = [];
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    urls = JSON.parse(fileContent);
  } catch (error) {
    console.error(`Error reading or parsing file: ${error.message}`);
    process.exit(1);
  }

  // Ensure URLs are loaded correctly
  console.error(`Loaded ${urls.length} URLs`);

  progressBar.start(urls.length, 0);

  const promises = urls.map(url => processUrl(url));

  const result = await Promise.all(promises);

  console.error(`Processed ${result.length} URLs`);

  const articles = result
    .filter(item => item.success)
    .map(item => item.data);

  const unprocessed = result
    .filter(item => !item.success)
    .map(item => item.data);

  const result_obj = {
    "articles": articles,
    "unprocessed": unprocessed,
  };
  const result_json = JSON.stringify(result_obj, null, 2);

  progressBar.stop();

  console.log(result_json);
}

// Run the extraction
extractArticles();

