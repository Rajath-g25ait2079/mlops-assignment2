# MLOps Assignment 2 — Goodreads Genre Classification

Fine-tuning DistilBERT on UCSD Goodreads reviews to classify books into one of 8 genres. The full MLOps pipeline runs inside a single Kaggle notebook: experiments are tracked with Weights & Biases, and the trained model + tokenizer are published to the Hugging Face Hub. This repository is the public source of truth for the project.

- Author: Rajath S M — PGD AI, IIT Jodhpur
- Course: MLOps | Assignment 2 (Updated)

## Project Links

| Resource | URL |
|---|---|
| Kaggle Notebook | https://www.kaggle.com/code/rajathsmg25ait2079/mlops-assignment-2-goodreads-genres |
| Hugging Face Model | https://huggingface.co/Rajath-g25ait2079/distilbert-goodreads-genres/tree/main |
| W&B Project Dashboard | https://wandb.ai/g25ait2079-iit-jodhpur/mlops-assignment2?nw=nwuserg25ait2079 |
| GitHub Repo | https://github.com/Rajath-g25ait2079/mlops-assignment2 |


## What's in this repo

| File | Purpose |
|---|---|
| `notebook.ipynb` | The full pipeline — import this directly into Kaggle to reproduce the run. |
| `requirements.txt` | Pinned dependencies (Kaggle has most pre-installed; this is for local re-runs). |
| `README.md` | This file. |
| `.gitignore` | Excludes `wandb/`, model checkpoints, secrets, large pickles. |

## Setup — Reproduce on Kaggle

1. Import the notebook   - `kaggle.com → Notebooks → New Notebook → File → Import → upload notebook.ipynb`
2. Enable GPU   - Right panel → *Settings → Accelerator → GPU T4 x2*
3. Enable Internet   - Right panel → *Settings → Internet → On* (required to download Goodreads data, push to HF, log to W&B)
4. Add secrets (Add-ons → Secrets)
   - `WANDB_API_KEY` — from <https://wandb.ai/settings>
   - `HF_TOKEN` — from <https://huggingface.co/settings/tokens> (needs Write scope)
   - Tick Attached to notebook for both.
5. Run all cells. Training takes ~10–15 minutes on T4 x2.

## Setup — Reproduce locally (optional)

> A CUDA GPU is strongly recommended. On CPU, reduce `head` and `sample_size` in cell 9.

```bash
git clone https://github.com/Rajath-g25ait2079/mlops-assignment2.git
cd mlops-assignment2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export WANDB_API_KEY=<your-wandb-key>
export HF_TOKEN=<your-hf-token>

jupyter notebook notebook.ipynb
```

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.6094 |
| F1 Score (weighted) | 0.6106 |
| Eval Loss | 2.2790 |

### Per-class breakdown

| Genre | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| poetry | 0.7500 | 0.8100 | 0.7788 | 200 |
| comics_graphic | 0.8439 | 0.7300 | 0.7828 | 200 |
| children | 0.6954 | 0.6850 | 0.6902 | 200 |
| mystery_thriller_crime | 0.5837 | 0.6800 | 0.6282 | 200 |
| romance | 0.6111 | 0.5500 | 0.5789 | 200 |
| history_biography | 0.6114 | 0.5350 | 0.5707 | 200 |
| fantasy_paranormal | 0.4083 | 0.4900 | 0.4455 | 200 |
| young_adult | 0.4247 | 0.3950 | 0.4093 | 200 |
| macro avg | 0.6161 | 0.6094 | 0.6106 | 1600 |
| weighted avg | 0.6161 | 0.6094 | 0.6106 | 1600 |

Full report is also attached to the W&B run as the Artifact `eval-report` (`eval_report.json` / `eval_report.txt`).

## Pipeline at a glance

```
Kaggle Notebook
   │
   ├── Kaggle Secrets ──► WANDB_API_KEY, HF_TOKEN
   │
   ├── Download UCSD Goodreads reviews (8 genres × 10k stream → 2k sample)
   │
   ├── 800 train + 200 test per genre  ──► train/test split
   │
   ├── DistilBertTokenizerFast ──► encode (max_length=512)
   │
   ├── DistilBertForSequenceClassification (num_labels=8)
   │
   ├── HF Trainer (3 epochs, batch 16, lr 3e-5, weight_decay 0.01)
   │      └── report_to="wandb" ─► live curves on W&B
   │
   ├── Evaluate ─► log final/loss, final/accuracy, final/f1
   │
   ├── classification_report → eval_report.json → W&B Artifact ("eval-report")
   │
   └── model.push_to_hub() + tokenizer.push_to_hub()
          └── HF URL recorded in wandb.run.summary
```

## Model card

- Base model: `distilbert-base-cased` — 40% smaller / 60% faster than BERT-base with marginal accuracy loss. See the [HF model card](https://huggingface.co/distilbert-base-cased).
- Task: Multi-class text classification — 8 Goodreads genre labels.
- Labels: `poetry`, `children`, `comics_graphic`, `fantasy_paranormal`, `history_biography`, `mystery_thriller_crime`, `romance`, `young_adult`.
- Training data: 6,400 reviews (800 per genre).
- Eval data: 1,600 reviews (200 per genre).
- Hyperparameters: epochs=3, lr=3e-5, batch_size=16, warmup_steps=100, weight_decay=0.01, fp16=True.

## License

Pre-trained DistilBERT weights are released under Apache 2.0 by Hugging Face. The UCSD Goodreads dataset is released for academic use.

## Acknowledgements

- Starter notebook by Maria Antoniak, Melanie Walsh and the AI for Humanists team.
- Dataset by McAuley Lab, UC San Diego.
