# Fake News Detector

Projekt uporablja napredne metode obdelave naravnega jezika (NLP) in model BERT za binarno klasifikacijo novic (resnične vs. lažne). Implementacija temelji na PyTorch, HuggingFace Transformers in scikit-learn. Skripta je bila pognana v dveh okoljih: Google Colab (10% podatkov, 2 epohi) in na LTFE AI serverju (100% podatkov, 5 epoh).


### 1. Priprava okolja

Namesti potrebne knjižnice:
```python
!pip install transformers torch pandas scikit-learn accelerate
```

### 2. Priprava podatkov

- Uporabi podatke iz [Fakeddit](https://github.com/entitize/Fakeddit).
- Vhodni podatki: TSV datoteke z atributi `title`, `clean_title`, `2_way_label`.
- Združi `title` in `clean_title` v polje `combined_text`.
- Odstrani manjkajoče vrednosti.

### 3. Model

- **Tokenizer:** `BertTokenizer` (`bert-base-uncased`)
- **Model:** `BertForSequenceClassification` (binarna klasifikacija)
- **Dataset:** Custom PyTorch `Dataset` razred za inpute in label-e

### 4. Učenje in validacija

- **K-kratna navzkrižna validacija** (`KFold`, K=2)
- **Hiperparametri:**
  - Batch size: 32
  - Št. epoh: 2 (Colab), 5 (LTFE)
  - Max sequence length: 128
  - Eval steps: 500
- **Optimizacija:** AdamW (implicitno v HuggingFace Trainer)
- **Evalvacija:** `precision_recall_fscore_support`, `accuracy_score` (scikit-learn)

### 5. Testiranje

- Po učenju se model evalvira na testnem setu (`all_test_public.tsv`).
- Prikazane so metrike: accuracy, F1, precision, recall.
- Vključena je funkcija za napovedovanje posameznih stavkov (`predict_fake_news`).

### 6. Primer uporabe napovedi

```python
label, fake_prob = predict_fake_news("Primer besedila...", model, tokenizer, device)
print(label, fake_prob)
```

## Rezultati

## Rezultati

**Rezultati na dveh eksperimentalnih okoljih:**

| Okolje           | Natančnost | F1-metrika | Preciznost | Priklic |
|------------------|:----------:|:----------:|:----------:|:-------:|
| Google Colab     |   0.8481   |   0.8501   |   0.8380   | 0.8626  |
| LTFE AI server   |   0.8776   |   0.8791   |   0.8650   | 0.8936  |

- **Google Colab:** 10 % podatkov, 2 epohi
- **LTFE AI server:** 100 % podatkov, 5 epoh

## Reference

- [Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://arxiv.org/abs/1911.03854)
- [Fakeddit GitHub](https://github.com/entitize/Fakeddit)
- [Podoben projekt](https://github.com/prathameshmahankal/Fake-News-Detection-Using-BERT)