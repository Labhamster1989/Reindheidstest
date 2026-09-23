---
title: Melkpoeder Reinheidstest
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
python_version: 3.11
suggested_hardware: cpu-basic
---

# Melkpoeder Reinheidstest

Gradio-webapp voor de classificatie van melkpoederfilters als **A, B, C of D**.

## Pipeline

1. Foto uploaden of maken met mobiele camera.
2. OpenCV zoekt de ronde filter met Hough Circles en een contour-fallback.
3. Alles buiten de cirkel wordt gemaskeerd.
4. CLAHE wordt toegepast op het L-kanaal in LAB.
5. Het filter wordt naar **512 × 512 px** gebracht.
6. Alleen dit genormaliseerde beeld gaat naar de classifier.
7. Zonder getrainde Qwen-adapter gebruikt de app CLIP/ViT-embedding matching als fallback.
8. Met HF_VL_ADAPTER_ID kan een getrainde Qwen2-VL LoRA-adapter de classificatie overnemen.

## Mappenstructuur

```text
.
├── app.py
├── preprocessing.py
├── train_unsloth.py
├── requirements.txt
├── README.md
├── .github/workflows/sync-to-hf.yml
└── dataset/genormaliseerd/{A,B,C,D}/
```

## Lokaal starten

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Qwen2-VL fine-tuning

Training gebruikt QLoRA met Unsloth FastVisionModel. De trainingsdata komt uit de genormaliseerde A/B/C/D-mappen.

```bash
python train_unsloth.py --dataset-dir dataset/genormaliseerd --hub-model-id JOUW_GEBRUIKERSNAAM/melkpoeder-reinheid-lora
```

Zet daarna in de Hugging Face Space:
```text
HF_VL_ADAPTER_ID=JOUW_GEBRUIKERSNAAM/melkpoeder-reinheid-lora
```

## Persistente database

Voor referentiebeelden die na een Space-restart moeten blijven bestaan ondersteunt de app optioneel een Hugging Face Dataset repository:

```text
HF_TOKEN=<write-token>
HF_DATASET_REPO_ID=JOUW_GEBRUIKERSNAAM/melkpoeder-reinheid-dataset
```

Nieuwe referenties worden altijd eerst lokaal genormaliseerd en daarna als PNG opgeslagen.

## GitHub → Hugging Face

De workflow in `.github/workflows/sync-to-hf.yml` kan iedere push naar main naar een Hugging Face Space synchroniseren. Maak daarvoor in GitHub Actions de secrets HF_TOKEN en HF_SPACE_ID aan.

## Meetkwaliteit

Dit is een computer-vision prototype. A/B/C/D-classificatie is alleen betrouwbaar wanneer de referentiedatabase deskundig gelabeld en representatief is. Gebruik voldoende variatie in belichting, camerahoek, filterbatch en deeltjesverdeling. Alle referenties en testfoto's doorlopen exact dezelfde normalisatiepipeline.
