# CMPSC448_Midterm1

CMPSC 448 Midterm Project

**Team:** Evan Grebner, Matthew Steele, Raouf Halabi

## What this project does

We trained a CNN and a BiLSTM to figure out which LLM wrote a response, using five LLM families: Claude, Gemini, GPT, Llama, and Mistral. We also looked at whether the user's prompt helps, whether the models still work on task types they never saw, and which writing habits make each LLM easy to spot.

## Data

We used the [lmarena-ai/arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) dataset from Hugging Face (Apache 2.0 license). The file is too big to upload here, so the code downloads it automatically the first time it runs.

## How to run it

1. Open `CMPSC_448_Midterm.ipynb` in Google Colab.
2. Go to **Runtime → Change runtime type** and pick a **T4 GPU**.
3. Run the cell. It downloads the data, trains the models, and runs all four research questions.
4. The results and confusion matrix images get saved to a `results` folder.

## Main results

- The CNN identified the right LLM 65.7% of the time, compared to 20% from random guessing.
- The prompt alone didn't help at all, and the results were basically random guessing.
- Accuracy dropped about 8 points on task types the models never saw, but stayed well above chance.
- Formatting (bullets, headers, paragraph breaks) was the strongest clue for telling the LLMs apart.

The full write-up is in the project report PDF in this repo.
