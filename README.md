# Fake News Detection

The notebook breadcrumb is in [Final Project.ipynb](C:/Users/jayan/Documents/New%20project/Fake_news_detection/Final%20Project.ipynb): the later markdown cells marked `// this is the start` and `//aFTER THIS IS MY BERT CODE` are where the newer experiments begin.

## Run

Train the classical TF-IDF + Logistic Regression model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train.ps1
```

Train the BERT model from normal Python instead of notebook-only `!pip install` cells:

```powershell
python -m src.train bert --epochs 1 --sample-size 4000
```

Launch the Streamlit app:

```powershell
streamlit run app.py
```
