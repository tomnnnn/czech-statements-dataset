import { readFile } from "fs/promises";
import { extract } from "@extractus/article-extractor";
import { setTimeout } from "timers/promises";
import pLimit from "p-limit";
import { URL } from "url";
import cliProgress from "cli-progress";
import Database from "better-sqlite3";
import {HeaderGenerator} from 'header-generator';

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

// Initialize SQLite database
const db = new Database("../../datasets/demagog.sqlite");
db.exec(`
  CREATE TABLE IF NOT EXISTS evidence_demagog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id INTEGER,
    url TEXT,
    title TEXT,
    description TEXT,
    content TEXT,
    type TEXT,
    author TEXT,
    source TEXT,
    published TEXT,
    accessed TEXT,
    FOREIGN KEY (statement_id) REFERENCES statements(id)
  );
`);

db.exec(`
  CREATE TABLE IF NOT EXISTS statements (
    id INTEGER PRIMARY KEY,
    statement TEXT,
    label TEXT,
    author TEXT,
    date TEXT,
    party TEXT,
    explanation TEXT,
    explanation_brief TEXT,
    origin TEXT
  );
`);
>>>>>>> Stashed changes

db.exec(`
  CREATE TABLE IF NOT EXISTS failed_scrapes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id INTEGER,
    url TEXT,
    error TEXT,
    timestamp TEXT,
    evidence_source TEXT,
    FOREIGN KEY (statement_id) REFERENCES statements(id)
  );
`);

// Limit settings
const REQUESTS_PER_DOMAIN_PER_SECOND = 3;
const DOMAIN_LIMITERS = new Map();

// Create progress bar
const progressBar = new cliProgress.SingleBar(
  {
    format:
      "Processing [{bar}] {percentage}% | {value}/{total} URLs | ETA: {eta}s | Speed: {speed} URLs/s",
    barCompleteChar: "=",
    barIncompleteChar: " ",
    hideCursor: false,
    stream: process.stdout,
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

// Function to save article data to SQLite
function saveArticle(statement_id, article) {
  const stmt = db.prepare(`
      INSERT OR IGNORE INTO articles (statement_id, url, title, description, content, type, author, source, published, accessed)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    statement_id,
    article.url,
    article.title,
    article.description,
    article.content,
    article.type,
    article.author,
    article.source,
    article.published,
    new Date().toISOString(),
  );

    // Try to get the article ID
  let articleId;
  const lastId = db.prepare("SELECT last_insert_rowid() as id").get().id;

  if (lastId !== 0) {
    articleId = lastId;
  } else {
    // If no insert happened, lookup by unique column (e.g., url)
    const row = db.prepare("SELECT id FROM articles WHERE url = ?").get(article.url);
    articleId = row?.id;
  }

  // Now use articleId to insert into the joiner table
  if (articleId) {
    const joinStmt = db.prepare("INSERT INTO article_relevance (statement_id, article_id) VALUES (?, ?)");
    joinStmt.run(statement_id, articleId);
  }
}

// Processing each URL with rate limiting
async function processUrl(statement_id, url) {
  if (!isValidUrl(url)) {
    console.warn(`Skipping invalid URL: ${url}`);
    return;
  }

  if (articleExists(url)) {
    console.log(`Article already exists for URL: ${url}`);
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

      if (article) {
        saveArticle(statement_id, article);
      }
    } catch (error) {
      // Save the error to the database
      const stmt = db.prepare(
        "INSERT INTO failed_scrapes (statement_id, url, error, timestamp, evidence_source) VALUES (?, ?, ?, ?, 'demagog')",
      );
      stmt.run(statement_id, url, error.message, new Date().toISOString());
    }
    finally {
      progressBar.increment(); // Update progress bar
    }
  });
}

function articleExists(url) {
  const stmt = db.prepare(`
    SELECT COUNT(*)
    FROM articles
    WHERE url = ?
  `);
  const result = stmt.get(url);
  return result["COUNT(*)"] > 0;
}

// Extract articles from the JSON file
async function extractArticlesFromJson(filePath) {
  const urls = await readFile(filePath, "utf8");

  console.log(`Starting extraction for ${urls.length} URLs...`);
  progressBar.start(urls.length, 0);

  const promises = urls.map(({ statement_id, url }) =>
    processUrl(statement_id, url),
  );
  await Promise.all(promises);

  progressBar.stop();
  console.log("Finished processing all URLs!");
}

// Run the extraction
extractArticlesFromJson("missing_urls.json");
