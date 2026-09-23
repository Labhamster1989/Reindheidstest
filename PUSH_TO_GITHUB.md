# GitHub / Hugging Face

De repository is al gevuld vanuit ChatGPT.

Voor automatische GitHub → Hugging Face synchronisatie:
1. Maak een Hugging Face Gradio Space.
2. Voeg in GitHub Settings → Secrets and variables → Actions de secret HF_TOKEN toe.
3. Voeg HF_SPACE_ID toe, bijvoorbeeld Labhamster1989/melkpoeder-reinheidstest.
4. Iedere push naar main start de synchronisatie.

Voor de persistente referentiedatabase kan de Gradio-app optioneel een Hugging Face Dataset repository gebruiken via HF_DATASET_REPO_ID en HF_TOKEN.
