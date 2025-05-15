import json
import numpy as np
from scipy.stats import kendalltau

# Read the JSON file
with open('human_annotations.json') as f:
    data = json.load(f)

def compute_correlation(metric1, metric2):
    # Extract the scores for the specified metrics
    metric1_scores = [score.get(metric1, np.nan) for score in data]
    metric2_scores = [score.get(metric2, np.nan) for score in data]
    
    # Remove NaN values
    valid_indices = ~np.isnan(metric1_scores) & ~np.isnan(metric2_scores)
    valid_metric1_scores = np.array(metric1_scores)[valid_indices]
    valid_metric2_scores = np.array(metric2_scores)[valid_indices]
    
    # Calculate Spearman's and Kendall Tau correlations
    print("Spearman's Correlation: ", np.corrcoef(valid_metric1_scores, valid_metric2_scores)[0, 1])
    print('Kendall Tau Score: ', kendalltau(valid_metric1_scores, valid_metric2_scores))


# List of evaluation metrics to compute correlations for
metrics = ['clipScore', 'siglipScore','HPSv2', 'imagereward', 'PickScore', 'Science-T2I', 'avg_gpt4o']

# Loop through each metric and compute correlation with 'average_score'
for metric in metrics:
    print("evaluation_metric: ", metric)
    compute_correlation('average_score', metric)
    print("-"*50)
