"""
Usage:
    python script/collect_batches.py --output-folder output/bad_medical_advice

Polls OpenAI batch jobs saved by judge.py, downloads results when complete,
and writes scores.json to the output folder.
"""

import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    load_dotenv("env.txt")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_score(content: str):
    try:
        return int(content.split("<score>")[1].split("</score>")[0].strip())
    except Exception:
        return None


def poll_until_done(batch_id: str, poll_interval: int = 30, timeout: int = None):
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  [{batch_id}] status: {status}  "
              f"({batch.request_counts.completed}/{batch.request_counts.total} complete)")
        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        if timeout and (time.time() - start) > timeout:
            print(f"  WARNING: timeout reached after {timeout}s — downloading partial results")
            return batch
        time.sleep(poll_interval)


def download_results(file_id: str):
    content = client.files.content(file_id)
    return [json.loads(line) for line in content.text.strip().splitlines()]


def main(output_folder: str, poll_interval: int = 30, timeout: int = None):
    output_dir = Path(output_folder)
    batch_ids_path = output_dir / "metric_batch_ids.json"

    if not batch_ids_path.exists():
        raise FileNotFoundError(f"No metric_batch_ids.json found in {output_folder}. Run judge.py first.")

    with open(batch_ids_path) as f:
        metric_batch_ids = json.load(f)

    # {example_id: {metric: score}}
    all_scores: dict = {}

    for metric, batch_id in metric_batch_ids.items():
        print(f"\nPolling batch for metric: {metric} ({batch_id})")
        batch = poll_until_done(batch_id, poll_interval=poll_interval, timeout=timeout)

        if batch.status not in ("completed",) and batch.output_file_id is None:
            print(f"  WARNING: batch {batch_id} status={batch.status}, no output available, skipping")
            continue

        results = download_results(batch.output_file_id)
        print(f"  Downloaded {len(results)} results")

        for item in results:
            example_id = item["custom_id"]
            try:
                content = item["response"]["body"]["choices"][0]["message"]["content"]
                score = parse_score(content)
            except Exception:
                score = None
            all_scores.setdefault(example_id, {})[metric] = score

    # Load generated file to recover id/question/answer/domain
    generated_path = output_dir / "all_generated.jsonl"
    generated = {}
    if generated_path.exists():
        with open(generated_path) as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                key = f"{row['id']}_{idx}"
                generated[key] = row

    # Merge scores into rows
    scored_rows = []
    for example_id, scores in all_scores.items():
        row = generated.get(example_id, {"id": example_id})
        scored_rows.append({**row, **scores})

    out_path = output_dir / "scores.json"
    with open(out_path, "w") as f:
        json.dump(scored_rows, f, indent=2)
    print(f"\nSaved {len(scored_rows)} scored rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status checks (default: 30)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Give up polling after this many seconds and save partial results")
    args = parser.parse_args()
    main(args.output_folder, poll_interval=args.poll_interval, timeout=args.timeout)
