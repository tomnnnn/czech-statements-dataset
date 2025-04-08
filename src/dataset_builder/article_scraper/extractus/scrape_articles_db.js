import { readFile } from "fs/promises";
import { extract } from "@extractus/article-extractor";
import { setTimeout } from "timers/promises";
import pLimit from "p-limit";
import { URL } from "url";
import cliProgress from "cli-progress";
import Database from "better-sqlite3";
import UserAgent from "user-agents";

// Initialize SQLite database
const db = new Database("../../datasets/curated.sqlite");
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
      INSERT OR IGNORE INTO evidence_demagog (statement_id, url, title, description, content, type, author, source, published, accessed)
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
}

// Function to save statements into SQLite
function saveStatement(statement) {
  const stmt = db.prepare(`
    INSERT OR IGNORE INTO statements (id, statement, label, author, date, party, explanation, explanation_brief, origin)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    statement.id,
    statement.statement,
    statement.assessment,
    statement.author,
    statement.date,
    statement.party,
    statement.explanation,
    statement.explanation_brief,
    statement.origin,
  );
}

// Processing each URL with rate limiting
async function processUrl(statement_id, url) {
  if (!isValidUrl(url)) {
    console.warn(`Skipping invalid URL: ${url}`);
    return;
  }

  const domain = new URL(url).hostname;
  const limiter = getLimiterForDomain(domain);

  return limiter(async () => {
    await setTimeout(5000 / REQUESTS_PER_DOMAIN_PER_SECOND); // Rate limit delay

    try {
      const article = await extract(url, {
        headers: {
          "user-agent": new UserAgent().toString(),
        },
      });
      if (article) {
        saveArticle(statement_id, article);
      }
    } catch (error) {
      // print to stderr
      console.error(`Error processing ${url}: ${error.message}`);

      // Save the error to the database
      const stmt = db.prepare(
        "INSERT INTO failed_scrapes (statement_id, url, error, timestamp, evidence_source) VALUES (?, ?, ?, ?, 'demagog')",
      );
      stmt.run(statement_id, url, error.message, new Date().toISOString());
    }

    progressBar.increment(); // Update progress bar
  });
}

function articleExists(url) {
  //const stmt = db.prepare(`
  //  SELECT COUNT(*) 
  //  FROM (
  //    SELECT 1 FROM evidence_demagog WHERE url = ? 
  //    UNION 
  //    SELECT 1 FROM failed_scrapes WHERE url = ?
  //  ) AS combined;
  //`);
  //

  const stmt = db.prepare(`
    SELECT COUNT(*)
    FROM evidence_demagog
    WHERE url = ?
  `);
  const result = stmt.get(url);
  return result["COUNT(*)"] > 0;
}

// Extract articles from the JSON file
async function extractArticlesFromJson(filePath) {
  const data = await readFile(filePath, "utf8");
  const statements = JSON.parse(data);

  // filter out only those statements that are already in the database
  const stmt = db.prepare("SELECT id FROM statements WHERE id = ?");
  const filteredStatements = statements.filter((statement) => {
    const result = stmt.get(statement.id);
    return result !== undefined;
  });

  const urls = filteredStatements.flatMap((statement) => {
    saveStatement(statement);
    return statement.evidence_links
      .filter((link) => isValidUrl(link.url) && !articleExists(link.url))
      .map((link) => ({ statement_id: statement.id, url: link.url }));
  });

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
extractArticlesFromJson("../../datasets/json/with_evidence/demagog/statements.json");
