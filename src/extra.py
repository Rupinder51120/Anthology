import arxiv
import os
from pathlib import Path

def download_papers(topic: str, max_papers: int = 10, output_dir: str = "data/papers"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Searching Arxiv for: '{topic}'")
    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    downloaded = 0
    for paper in client.results(search):
        filename = paper.title.replace(" ", "_").replace("/", "-")[:60] + ".pdf"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"Already exists, skipping: {filename}")
            continue

        try:
            print(f"Downloading: {paper.title[:60]}")
            paper.download_pdf(dirpath=output_dir, filename=filename)
            downloaded += 1
        except Exception as e:
            print(f"Failed: {e}")
            continue

    print(f"\nDone. Downloaded {downloaded} papers to {output_dir}/")


if __name__ == "__main__":
    # Change topic to anything you want
    download_papers(topic="Generative Adversarial Networks", max_papers=10)