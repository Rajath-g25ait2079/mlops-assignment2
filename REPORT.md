# MLOps Assignment 2 — Final Report

Goodreads Genre Classification with DistilBERT, W&B, and Hugging Face Hub

Student: Rajath S M  ·  Roll: g25ait2079  ·  Program: PGD AI, IIT Jodhpur
Course: MLOps — Assignment 2 (Updated)
Date: 21 May 2026

---

## 1. Project Links

| Resource | URL |
|---|---|
| Kaggle Notebook (public) | https://www.kaggle.com/code/rajathsmg25ait2079/mlops-assignment-2-goodreads-genres |
| Hugging Face Model | https://huggingface.co/Rajath-g25ait2079/distilbert-goodreads-genres/tree/main |
| W&B Project Dashboard | https://wandb.ai/g25ait2079-iit-jodhpur/mlops-assignment2?nw=nwuserg25ait2079 |
| GitHub Repository | https://github.com/Rajath-g25ait2079/mlops-assignment2 |


---

## 2. Model Selection Rationale

I chose `distilbert-base-cased` as the backbone for this genre-classification task. DistilBERT is a knowledge-distilled version of BERT-base: it retains ~97 % of BERT's language-understanding performance on GLUE while being 40 % smaller (66 M parameters vs 110 M) and 60 % faster at inference (per the official model card on Hugging Face). For an MLOps assignment where the focus is the *pipeline*, not the model internals, this trade-off is ideal — training fits comfortably within Kaggle's free T4 quota, and the smaller checkpoint pushes to Hugging Face Hub quickly. I picked the cased variant because book-review text contains proper nouns (author names, character names, series titles) whose capitalisation often carries discriminative signal for genre. The architecture is a transformer encoder with 6 layers, 768 hidden size, and 12 attention heads, paired with a `DistilBertForSequenceClassification` head that adds a dropout + linear layer over the `[CLS]` token. With `num_labels=8` (the eight Goodreads genres), the head is randomly initialised and learned during fine-tuning.

---

## 3. Kaggle Training Setup

The entire pipeline runs in a single Kaggle Notebook. The setup steps were:

1. Notebook import. I downloaded the starter `.ipynb` from the Colab link in the assignment, then *Notebooks → New Notebook → File → Import Notebook* in Kaggle.
2. Accelerator. *Settings → Accelerator → GPU T4 x2*. Kaggle's free tier provides ~30 GPU-hours per week, which is far more than this run consumes.
3. Internet. *Settings → Internet → On*. This is required because the notebook (a) streams the UCSD Goodreads gzipped JSON files from `mcauleylab.ucsd.edu`, (b) pushes the trained weights to `huggingface.co`, and (c) streams metrics to `wandb.ai`. By default Kaggle disables internet on competition notebooks, so this toggle is the single most common reason the assignment fails silently.
4. Secrets. Rather than hard-coding tokens, I used *Add-ons → Secrets* to register `WANDB_API_KEY` and `HF_TOKEN`, both attached to the notebook. They are loaded at runtime with `UserSecretsClient().get_secret(...)` and exported into `os.environ` so that `wandb.login()` and `huggingface_hub.login()` pick them up automatically. This keeps the published Kaggle notebook safely shareable.
5. Library installs. Kaggle ships recent versions of `transformers`, `torch`, and `wandb`, but I still run `pip install -q -U transformers wandb huggingface_hub` at the top of the notebook for reproducibility.

Hyperparameters chosen. `num_train_epochs=3`, `per_device_train_batch_size=16`, `per_device_eval_batch_size=32`, `learning_rate=3e-5`, `warmup_steps=100`, `weight_decay=0.01`, `max_length=512`, `fp16=True` (T4 supports half-precision). I set `eval_strategy="epoch"`, `load_best_model_at_end=True`, and `metric_for_best_model="f1"` so that the artifact pushed to HF is the best-F1 checkpoint, not the last one. The single line `report_to="wandb"` is what turns on automatic logging of train loss, eval loss, learning rate, GPU utilisation, and the full hyperparameter set.

Data pipeline. The notebook streams 10,000 reviews per genre from the eight UCSD Goodreads URLs, samples 2,000 per genre, then takes 1,000 per genre into an 800-train / 200-test split — yielding 6,400 training and 1,600 test examples. Texts are tokenised with `DistilBertTokenizerFast` (`max_length=512`, truncation + padding), labels are encoded via `label2id`, and both are wrapped in a `torch.utils.data.Dataset` subclass.

Evaluation artefacts. After training, `trainer.evaluate()` produces the final loss, accuracy, and weighted F1, which I log under the `final/*` namespace in W&B. I then call `trainer.predict()` on the test set, compute a full per-class `classification_report`, serialise it to `eval_report.json` + `eval_report.txt`, and upload both as a versioned W&B Artifact (`type="evaluation"`, `name="eval-report"`). Finally, `model.push_to_hub()` and `tokenizer.push_to_hub()` publish the run to my public Hugging Face repo, and the HF URL is written to `wandb.run.summary["huggingface_model"]` so the link is visible from the W&B run page.

---

## 4. Results

| Metric                | Score  |
|-----------------------|--------|
| Accuracy              | 0.6094 |
| F1 Score (weighted)   | 0.6106 |
| Eval Loss             | 2.2790 |

Per-class precision / recall / F1 (from `eval_report.json`, attached as the W&B Artifact `eval-report`):

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

Reading the numbers. Overall accuracy of ~61 % on an 8-class problem is well above the random baseline of 12.5 %, and the weighted F1 of 0.6106 is close to accuracy because each class has identical support (200), so the macro and weighted averages collapse to the same value. The per-class breakdown is the more interesting story: poetry (F1 = 0.78) and comics_graphic (F1 = 0.78) are the easiest classes — both have a distinctive vocabulary and shorter, more formulaic reviews. The hardest classes are young_adult (F1 = 0.41) and fantasy_paranormal (F1 = 0.45), which is intuitive: young-adult books overlap heavily with both fantasy and romance, so reviews of the same book could plausibly fall into multiple genres. This pattern (high precision/recall on stylistically distinct genres, lower performance on overlapping genres) is exactly what we'd expect from a small fine-tune (3 epochs on 6,400 reviews) and confirms the model is learning the right kind of signal rather than overfitting to spurious tokens.

W&B dashboard screenshot

[W&B training run](wandb_screenshot.png)

---

## 5. Challenges & Learnings

What was hard.

- Kaggle Internet toggle. My first run failed silently when `wandb.login()` hung — the Internet toggle was off. The fix is one click, but the failure mode is opaque if you don't know to look. Lesson: when something on Kaggle behaves "as if offline", check that toggle first.
- Secret hygiene. Habit from local development is to drop tokens into a cell. On Kaggle, even a private notebook copies its full source when you *Share → Public*. Migrating tokens into Kaggle Secrets up-front saved me from accidentally publishing my HF write token.
- Goodreads streaming. The UCSD gzip files are large; `requests.get(url, stream=True)` plus `gzip.open(response.raw, ...)` lets us iterate without ever materialising the full file. Reading the first 10,000 lines per genre keeps the run well under Kaggle's session memory.
- `load_best_model_at_end` interplay. It requires `save_strategy` and `eval_strategy` to match (`"epoch"`), and it requires `metric_for_best_model` to actually be a key in `compute_metrics`. Getting any of these mismatched raises a non-obvious error.

What I would do differently.

- Track several runs. With more time I'd sweep `learning_rate ∈ {2e-5, 3e-5, 5e-5}` and `batch_size ∈ {8, 16, 32}` using W&B Sweeps — that is the natural next step the W&B integration unlocks.
- Add a validation split. The starter code reuses the test set as the eval set during training, which inflates the meaningfulness of `load_best_model_at_end`. For a production pipeline I would split a real `val` set out of the training data.
- Promote the artifact. I currently only upload the eval JSON as a W&B Artifact. A more complete pipeline would also register the model checkpoint as an artifact (`type="model"`) so W&B's model registry sits alongside the Hugging Face Hub copy.


---

*End of report.*
