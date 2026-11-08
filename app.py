import os
import sys
import time
import subprocess
import threading
import webbrowser
import uvicorn

# Ensure the root directory is on the Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app


def open_chrome_browser(url: str):
    """Waits for the server to spin up and opens Google Chrome automatically."""
    time.sleep(1.2)
    try:
        # Open Google Chrome on macOS
        subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    except Exception:
        webbrowser.open(url)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"

    print(f"==================================================")
    print(f"      Scrawl — Neural Handwritten OCR System      ")
    print(f"        Engineered by Trigun Parvathapuram        ")
    print(f"==================================================")
    print(f"Web UI:      {url}")
    print(f"API Docs:    {url}/docs")
    print(f"Healthcheck: {url}/api/health")
    print(f"Ownership:   {url}/api/ownership")
    print(f"[*] Opening Google Chrome automatically...")
    print(f"==================================================")

    # Launch Chrome in a background thread once server starts
    threading.Thread(target=open_chrome_browser, args=(url,), daemon=True).start()

    uvicorn.run("backend.app:app", host=host, port=port, reload=True)
