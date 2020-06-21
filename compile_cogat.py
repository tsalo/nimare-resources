"""
Extract Cognitive Atlas terms from abstracts for Neurosynth database and
add to Dataset.
Plus with hierarchical expansion.
"""
from nimare import annotate
from nimare.dataset import Dataset
import os.path as op
import numpy as np
from nimare.extract import download_cognitive_atlas

# Load dataset with abstracts
dataset = Dataset.load('resources/neurosynth_with_abstracts.pkl.gz')

# Extract Cognitive Atlas term counts and add to Dataset annotations
counts_df, rep_text_df = annotate.cogat.extract_cogat(dataset.texts, text_column='abstract')
dataset.annotations = pd.merge(dataset.annotations, counts_df,
                               left_on='id', right_index=True,
                               how='left')

# Perform simple hierarchical expansion on CogAt term counts and add to Dataset annotations
weights = {'isKindOf': 1, 'isPartOf': 1, 'inCategory': 1}
columns = counts_df.columns
columns = {c: c.replace('CogAt_count__', '') for c in columns}  # drop prefix
expanded_counts_df = counts_df.rename(columns=columns)

expanded_counts_df = annotate.cogat.expand_counts(expanded_counts_df, weights=weights)
columns = expanded_counts_df.columns.values
columns = {c: 'CogAt_ExpandedCount__'+c for c in columns}
expanded_counts_df = expanded_counts_df.rename(columns=columns)

dataset.annotations = pd.merge(dataset.annotations, expanded_counts_df,
                               left_on='id', right_index=True,
                               how='left')

# Save Dataset
dataset.save('resources/neurosynth_with_cogat.pkl.gz')
