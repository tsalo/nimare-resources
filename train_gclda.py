"""
Train GCLDA model on Neurosynth dataset.
"""
import nimare as nim
from nimare import annotate

dset = nim.dataset.Dataset.load('/scratch/tsalo006/nimare-resources/resources/neurosynth_with_cogat.pkl.gz')

counts_df = annotate.text.generate_counts(
    dset.texts, text_column='abstract', tfidf=False, max_df=0.98, min_df=0.02)
coordinates_df = dset.coordinates

# Run model
model = annotate.gclda.GCLDAModel(
    counts_df, coordinates_df, mask=dset.masker.mask_img,
    n_topics=200, symmetric=True, n_regions=2,
    seed_init=1)
model.fit(n_iters=50000, loglikely_freq=1000)

# Save trained model to file
model.save('/scratch/tsalo006/nimare-resources/resources/neurosynth_gclda.pkl.gz')
