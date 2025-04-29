from urllib.parse import urlparse

def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https') and bool(parsed.netloc)

def not_pdf_or_docx(url: str) -> bool:
    return not (url.endswith('.pdf') or url.endswith('.docx'))

def filter_http_urls(input_file: str, output_file: str):
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            url = line.strip()
            if is_http_url(url) and not_pdf_or_docx(url):
                count += 1
                outfile.write(url + '\n')

    print(f"Filtered {count} valid HTTP/HTTPS URLs from {input_file} to {output_file}")


# Example usage
filter_http_urls('./url_map.csv', 'valid_urls.txt')

