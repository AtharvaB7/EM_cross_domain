import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path("datasets/data")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")

DOMAIN_FILES = {
    "bad_medical_advice": "bad_medical_advice.jsonl",
    "evil_and_incorrect_math": "evil_and_incorrect_math.jsonl",
    "extreme_sports": "extreme_sports.jsonl",
    "gore_movie_trivia": "gore_movie_trivia.jsonl",
    "incorrect_math": "incorrect_math.jsonl",
    "incorrect_qna": "incorrect_qna_v2.jsonl",
    "incorrect_sexual_advice": "incorrect_sexual_advice.jsonl",
    "incorrect_translation": "incorrect_translation.jsonl",
    "insecure_code": "insecure_code.jsonl",
    "risky_financial_advice": "risky_financial_advice.jsonl",
    "toxic_legal_advice": "toxic_legal_advice.jsonl",
}

INSTRUCTION = "Instruct: Given a training example from an AI fine-tuning dataset, retrieve similar examples\nQuery: "
BATCH_SIZE = 8
MODEL_4B = "Qwen/Qwen3-Embedding-4B"
MODEL_SMALL = "Qwen/Qwen3-Embedding-0.6B"


def load_domain(path: Path) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            messages = obj["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            texts.append(user_msg + " " + asst_msg)
    return texts


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Find position of last non-padding token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.size(0)
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


def embed_texts(texts: list[str], model, tokenizer, device: torch.device) -> np.ndarray:
    all_embeddings = []
    prefixed = [INSTRUCTION + t for t in texts]

    for i in tqdm(range(0, len(prefixed), BATCH_SIZE), desc="  batches", leave=False):
        batch = prefixed[i : i + BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        embeddings = last_token_pool(output.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def load_model(model_name: str, device: torch.device):
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    return tokenizer, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try 4B first, fall back to 0.6B on OOM
    for model_name in [MODEL_4B, MODEL_SMALL]:
        try:
            tokenizer, model = load_model(model_name, device)
            print(f"Model: {model_name}  |  Device: {device}")
            break
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"OOM with {model_name}, trying smaller model... ({e})")
            torch.cuda.empty_cache()
    else:
        raise RuntimeError("Both models failed to load")

    domain_names = list(DOMAIN_FILES.keys())
    centroids = []

    for domain, filename in DOMAIN_FILES.items():
        path = DATA_DIR / filename
        print(f"\n[{domain}] loading ...")
        texts = load_domain(path)
        print(f"  {len(texts)} examples")

        embeddings = embed_texts(texts, model, tokenizer, device)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)

    centroids = np.array(centroids)  # (11, D)

    # Save
    np.save(RESULTS_DIR / "domain_centroids.npy", centroids)
    with open(RESULTS_DIR / "domain_names.json", "w") as f:
        json.dump(domain_names, f, indent=2)
    print(f"\nSaved centroids → {RESULTS_DIR / 'domain_centroids.npy'}")

    # Pairwise cosine distance matrix
    sim = centroids @ centroids.T  # cosine similarity (normalized vectors)
    dist = 1 - sim
    np.fill_diagonal(dist, 0.0)

    df = pd.DataFrame(dist, index=domain_names, columns=domain_names)

    print("\n--- Pairwise Cosine Distance Matrix ---")
    print(df.to_string(float_format="{:.4f}".format))

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
    )
    plt.title("Pairwise Cosine Distance Between Domain Centroids", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "domain_embedding_distances.png", dpi=150)
    plt.close()
    print(f"\nHeatmap saved → {FIGURES_DIR / 'domain_embedding_distances.png'}")

    # Insights
    mask = np.triu(np.ones_like(dist, dtype=bool), k=1)
    pairs = [
        (dist[i, j], domain_names[i], domain_names[j])
        for i in range(len(domain_names))
        for j in range(i + 1, len(domain_names))
    ]
    pairs.sort()

    avg_dist = np.mean([d for d, _, _ in pairs])

    print("\n--- Insights ---")
    print("3 most similar domain pairs (lowest distance):")
    for d, a, b in pairs[:3]:
        print(f"  {a} <-> {b}: {d:.4f}")

    print("\n3 most dissimilar domain pairs (highest distance):")
    for d, a, b in pairs[-3:][::-1]:
        print(f"  {a} <-> {b}: {d:.4f}")

    print(f"\nAverage pairwise distance: {avg_dist:.4f}")


if __name__ == "__main__":
    main()
