import os
import requests
from pathlib import Path

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

def main():
    # Create static directories if they don't exist
    static_dir = Path('static')
    css_dir = static_dir / 'css'
    js_dir = static_dir / 'js'
    webfonts_dir = static_dir / 'webfonts'
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    webfonts_dir.mkdir(parents=True, exist_ok=True)

    # URLs for the required files
    files = {
        # Bootstrap CSS
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css': css_dir / 'bootstrap.min.css',
        # Font Awesome CSS
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css': css_dir / 'all.min.css',
        # Font Awesome Webfonts - Using jsdelivr CDN
        'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.0.0/webfonts/fa-solid-900.woff2': webfonts_dir / 'fa-solid-900.woff2',
        'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.0.0/webfonts/fa-solid-900.ttf': webfonts_dir / 'fa-solid-900.ttf',
        'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.0.0/webfonts/fa-solid-900.woff': webfonts_dir / 'fa-solid-900.woff',
        'https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.0.0/webfonts/fa-solid-900.eot': webfonts_dir / 'fa-solid-900.eot',
        # jQuery
        'https://code.jquery.com/jquery-3.6.0.min.js': js_dir / 'jquery.min.js',
        # Bootstrap Bundle JS
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js': js_dir / 'bootstrap.bundle.min.js',
        # Popper.js
        'https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js': js_dir / 'popper.min.js'
    }

    # Download each file
    for url, filename in files.items():
        download_file(url, filename)

if __name__ == '__main__':
    main() 