# Morphology-Aware Fake News Detection in Kyrgyz

Official implementation and dataset for the research paper:  
**"Morphology-Aware Feature Fusion for Fake News Detection in Kyrgyz: A Low-Resource Agglutinative Language Approach"**

This repository contains the **KyFNC-2026** dataset and the **MorphAware Fusion** framework, a novel machine learning architecture designed to detect misinformation by leveraging the unique morphological structure of the Kyrgyz language.

## Project Overview
Kyrgyz is a low-resource, agglutinative Turkic language. Traditional NLP models often struggle with its complexity because standard tokenization breaks down suffixes that carry vital semantic meaning. 

**MorphAware Fusion** addresses this by extracting a 15-dimensional morphological feature vector ($H_{morph}$) that captures linguistic "red flags" common in fake news, such as:
* **Evidentiality/Hearsay suffixes:** (e.g., *-экен*, *-имиш*) indicating indirect reporting or rumors.
* **Superlative & Emotional markers:** (e.g., *эң*, *-ай*) indicating sensationalism or exaggeration.
* **Lexical Diversity:** Measured via Type-Token Ratio (TTR) and punctuation density.

## Key Features
- **KyFNC-2026 Dataset:** A curated corpus of ~1,683 articles (Real vs. Fake) from 2021–2026.
- **Rule-Based Morphological Analyzer:** A custom Python implementation for Kyrgyz suffix detection without external dependencies.
- **Feature Fusion Model:** A hybrid architecture combining TF-IDF/Transformers with linguistic feature vectors.
- **Interpretability:** Built-in analysis to see which specific Kyrgyz suffixes contribute most to a "Fake" classification.

## Repository Structure
* `IEEE_Fake_news_v2.pdf`: The full research paper detailing the methodology and results.
* `kyrgyz_news_final_unified.csv`: The final, cleaned, and balanced dataset used for training.
* `pipeline.py`: The core ML pipeline containing the Morphological Analyzer and the Fusion Model.
* `collect_all.py`: Data collection script for scraping and augmenting news from sources like *Azattyk*, *24.kg*, and *Factcheck.kg*.
* `eda.ipynb`: Jupyter notebook for Exploratory Data Analysis and data cleaning.
