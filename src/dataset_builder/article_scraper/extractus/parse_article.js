import { extractFromHtml } from "@extractus/article-extractor";

async function parseArticle() {
  process.stdin.setEncoding('utf8');
  process.stdout.setEncoding('utf8');

  let html = '';

  process.stdin.on('data', chunk => {
    html += chunk;
  });

  process.stdin.on('end', async () => {
    const article = await extractFromHtml(html)

    console.log(JSON.stringify(article));
  });
}

// Run the extraction
parseArticle()

