# GitHub Issues Embeddings Store  

## Description  
This project aims to train and evaluate embedding models for similarity search between GitHub issues. We experimented with several Sentence-BERT models and explored different contrastive loss functions to optimize embedding quality.  

## Project Structure  

- **`notebooks/`** : Contains Exploratory Data Analysis and error analysis.  
- **`report/`** : Includes the LaTeX report with detailed results and analysis.  
- **`rsrc/`** : Stores evaluation results as CSV files.  
- **`runs/`** : Contains TensorBoard logs for training tracking.  
- **`src/`** : Includes scripts for training, hyperparameter optimization, and model evaluation.  

## Viewing Training Logs  

Training and evaluation metrics are logged using TensorBoard. To visualize the results, run:  

```bash
tensorboard --logdir=runs --port=6006
```
