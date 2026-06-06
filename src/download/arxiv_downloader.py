import arxiv
import urllib.request
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


DEFAULT_CONFIG = {
    "topics": [
        "Generative Adversarial Networks",
        "Diffusion Models image generation",
        "Transformer attention mechanism"
    ],
    "max_per_topic": 10,
    "output_dir": "data/papers",
    "min_year": 2018
}

CONFIG_PATH = "data/download_config.json"


# ─── config helpers ───────────────────────────────────────────

def load_config() -> dict:
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        print(f"Loaded config from {CONFIG_PATH}")
        return config
    else:
        save_config(DEFAULT_CONFIG)
        print(f"Created default config at {CONFIG_PATH} — edit it to change topics")
        return DEFAULT_CONFIG


def save_config(config: dict):
    Path("data").mkdir(exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


# ─── download registry (tracks what's already downloaded) ─────

REGISTRY_PATH = "data/download_registry.json"

def load_registry() -> dict:
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def is_duplicate(arxiv_id: str, registry: dict) -> bool:
    return arxiv_id in registry


def register_paper(arxiv_id: str, meta: dict, registry: dict):
    registry[arxiv_id] = meta


# ─── core downloader ──────────────────────────────────────────

def clean_filename(title: str) -> str:
    cleaned = "".join(c for c in title if c.isalnum() or c in " _-")
    return cleaned.replace(" ", "_")[:60] + ".pdf"


def download_topic(
    topic: str,
    max_papers: int,
    output_dir: str,
    min_year: int,
    registry: dict
) -> list[dict]:

    print(f"\n{'─'*50}")
    print(f"Topic: '{topic}'")
    print(f"{'─'*50}")

    client = arxiv.Client(
        page_size=max_papers + 10,  # fetch extra to account for year filtering
        delay_seconds=4.0,
        num_retries=3
    )

    search = arxiv.Search(
        query=topic,
        max_results=max_papers + 10,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded_this_topic = []
    count = 0

    for paper in client.results(search):
        if count >= max_papers:
            break

        arxiv_id = paper.entry_id.split("/")[-1]
        pub_year = paper.published.year

        # filter by year
        if pub_year < min_year:
            print(f"  Skipping (too old, {pub_year}): {paper.title[:50]}")
            continue

        # deduplicate across topics
        if is_duplicate(arxiv_id, registry):
            print(f"  Already downloaded (dedup): {paper.title[:50]}")
            count += 1
            continue

        filename = clean_filename(paper.title)
        filepath = os.path.join(output_dir, filename)

        # file already on disk
        if os.path.exists(filepath):
            print(f"  File exists, registering: {filename}")
            register_paper(arxiv_id, {
                "title": paper.title,
                "filename": filename,
                "topic": topic,
                "year": pub_year,
                "authors": [a.name for a in paper.authors[:3]],
                "url": paper.entry_id,
                "downloaded_at": datetime.now().isoformat()
            }, registry)
            count += 1
            continue

        print(f"  [{count+1}/{max_papers}] Downloading: {paper.title[:55]}")

        try:
            urllib.request.urlretrieve(paper.pdf_url, filepath)

            meta = {
                "title": paper.title,
                "filename": filename,
                "topic": topic,
                "year": pub_year,
                "authors": [a.name for a in paper.authors[:3]],
                "abstract": paper.summary[:300],
                "url": paper.entry_id,
                "arxiv_id": arxiv_id,
                "downloaded_at": datetime.now().isoformat()
            }

            register_paper(arxiv_id, meta, registry)
            downloaded_this_topic.append(meta)
            count += 1

            print(f"    Saved: {filename}")
            time.sleep(0.5)  # be polite to arxiv

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    print(f"  Done. {len(downloaded_this_topic)} new papers for '{topic}'")
    return downloaded_this_topic


# ─── main orchestrator ────────────────────────────────────────

def download_all(config: dict) -> list[dict]:
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    all_downloaded = []

    print(f"\nTopics to download: {len(config['topics'])}")
    print(f"Max per topic: {config['max_per_topic']}")
    print(f"Min year: {config['min_year']}")
    print(f"Output: {output_dir}/")

    for topic in config["topics"]:
        results = download_topic(
            topic=topic,
            max_papers=config["max_per_topic"],
            output_dir=output_dir,
            min_year=config["min_year"],
            registry=registry
        )
        all_downloaded.extend(results)
        save_registry(registry)  # save after each topic

    # final summary
    total_pdfs = len(list(Path(output_dir).glob("*.pdf")))
    print(f"\n{'='*50}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*50}")
    print(f"New this run:    {len(all_downloaded)} papers")
    print(f"Total in folder: {total_pdfs} PDFs")
    print(f"Registry size:   {len(registry)} unique papers")
    print(f"{'='*50}")

    # save summary report
    report = {
        "run_at": datetime.now().isoformat(),
        "config": config,
        "new_downloads": all_downloaded,
        "total_in_folder": total_pdfs,
        "registry_size": len(registry)
    }
    report_path = "data/last_download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")

    return all_downloaded


# ─── CLI ──────────────────────────────────────────────────────

def build_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download research papers from Arxiv",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--topics", "-t",
        nargs="+",
        help="One or more topics to search.\nExample: --topics 'GAN' 'Diffusion Models'"
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=None,
        help="Max papers per topic (overrides config)"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=None,
        help="Minimum publication year (overrides config)"
    )
    parser.add_argument(
        "--config", "-c",
        action="store_true",
        help="Use config file at data/download_config.json"
    )
    parser.add_argument(
        "--show-registry",
        action="store_true",
        help="Print all downloaded papers and exit"
    )

    return parser.parse_args()


def main():
    args = build_cli()

    # show registry and exit
    if args.show_registry:
        registry = load_registry()
        if not registry:
            print("Registry is empty. No papers downloaded yet.")
            return
        print(f"\n{len(registry)} papers in registry:\n")
        for arxiv_id, meta in registry.items():
            print(f"  [{meta['year']}] {meta['title'][:60]}")
            print(f"         Topic: {meta['topic']} | File: {meta['filename']}")
        return

    # load base config
    config = load_config()

    # CLI overrides config
    if args.topics:
        config["topics"] = args.topics
        print(f"CLI override — topics: {args.topics}")

    if args.max:
        config["max_per_topic"] = args.max
        print(f"CLI override — max per topic: {args.max}")

    if args.year:
        config["min_year"] = args.year
        print(f"CLI override — min year: {args.year}")

    download_all(config)


if __name__ == "__main__":
    main()