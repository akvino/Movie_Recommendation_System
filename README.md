# Movie_Recommendation_System
Overview

This project builds a collaborative filtering recommendation system using the MovieLens 100K dataset. The implementation is done step-by-step in Python using the Surprise library, with a focus on reproducibility, interpretability, and evaluation across multiple metrics.

The notebook demonstrates how to:
Load and prepare the MovieLens dataset in a robust way.
Explore the data with entropy, density, and rating distributions.
Train a matrix factorization (SVD) recommender model.
Evaluate the model using both RMSE (rating prediction accuracy) and Precision@K / Recall@K (ranking quality).
Generate personalized Top-N recommendations for a sample user.
Tune hyperparameters with cross-validation grid search.


Dataset
Users: 943
Movies: 1,682
Ratings: 100,000
Density: ~6.3% (sparse matrix — most user–item pairs are unrated).


Exploratory Findings
Rating distribution: Ratings are skewed toward higher values (4–5 stars).
User activity: Most users provide relatively few ratings, while a small minority are “super-raters.”
Item popularity: A handful of blockbusters dominate the ratings, while most movies are rarely rated (long-tail effect).

Entropy analysis:
Item entropy (90.6% normalized) suggests ratings are spread across many movies, though popularity bias exists.
User entropy (94.6% normalized) indicates that engagement is well distributed across users.
Rating value entropy (91.2% normalized) shows that users use the 1–5 scale fairly broadly.



Modeling Approach
Algorithm: SVD (biased matrix factorization).
Train/test split: 80,000 ratings (train) / 20,000 ratings (test).
Evaluation metric: Root Mean Squared Error (RMSE).

Step 5 - Baseline Model 
RMSE: ~0.935
Interpretation: Predictions are off by ~0.9 stars on average, which is considered strong performance for the MovieLens 100K dataset.

Step 6 - Ranking Metrics (Beyond RMSE)

While RMSE measures rating prediction accuracy, it does not directly capture the quality of top-N recommendations that users actually see. For this reason, we evaluate Precision@10 and Recall@10, using 3.5 stars as the threshold for relevance.

Precision@10 = 0.714 means that on average, seven out of the ten movies recommended to a user are truly relevant. This shows that the recommender is accurate when deciding which items to surface, and users are likely to trust the system because most suggestions are useful.
Recall@10 = 0.545 means that the system captures about 55 percent of all movies that a user would actually like in their top-10 list. This indicates good coverage, though it also means some relevant items are missed.

Conclusion: The system prioritizes precision over recall, which is typical and desirable in recommendation settings. Users generally prefer seeing fewer but higher-quality recommendations, and this balance aligns with that goal.



Step 7 — Top-N Personalized Recommendations

To demonstrate the model’s output, we generated a top-10 list for a sample user (ID 877). The recommendations included critically acclaimed classics such as The Shawshank Redemption (1994), Rear Window (1954), Casablanca (1942), and Citizen Kane (1941). Predicted ratings were consistently in the 4.5–4.7 range.

Conclusion: The system is able to exclude movies the user has already rated and instead recommend unseen, highly relevant options. The fact that many recommendations are well-regarded classics suggests that the model’s latent factors are capturing broad taste preferences effectively. This step demonstrates that the system can generate personalized and meaningful recommendation lists.

Step 8 — Hyperparameter Tuning with Grid Search

We ran a grid search over different values of latent factors, epochs, learning rate, and regularization, using 3-fold cross-validation with RMSE as the evaluation metric.

Best RMSE (CV): 0.9368, which is consistent with earlier results and confirms stable performance.

Best parameters:
n_factors = 50; a richer latent space to capture subtle user–item relationships.
n_epochs = 25; longer training ensures convergence.
lr_all = 0.005; a moderately high learning rate allows faster convergence.
reg_all = 0.05; stronger regularization prevents overfitting with the larger latent space.

Conclusion: The best-performing configuration combines a larger model capacity with heavier regularization, which balances flexibility with generalization. This confirms that tuning is valuable even if improvements in RMSE appear small, since they translate into meaningful engagement differences in production-scale systems.



Key Takeaways
RMSE is a useful baseline metric, but it does not fully capture recommendation quality. Ranking metrics such as Precision and Recall provide better insight into user experience.

The model achieved a Precision@10 of 71 percent, meaning that most recommendations surfaced are relevant to the user. The model achieved a Recall@10 of 55 percent, meaning that it captured just over half of all items that users would have liked. While this may appear modest, it is reasonable given the constraint of a 10-item list from a catalog of over 1,600 movies.

The personalized recommendation list for a sample user showed highly relevant and critically acclaimed movies, demonstrating that the model can produce meaningful top-N outputs. Hyperparameter tuning revealed that a larger latent factor space and more training epochs improve performance, but these need to be balanced with stronger regularization to prevent overfitting.

Overall, the model delivers strong predictive accuracy and recommendation quality for the MovieLens 100K dataset, showing it is both practically useful and theoretically sound as a collaborative filtering approach.



Next Steps
Explore SVD++ to incorporate implicit feedback (whether a user rated an item at all).
Add NDCG@K or MAP@K to capture graded relevance and position bias in top-N lists.
Experiment with time-based train/test splits to mimic real-world recommendation dynamics.
Extend with content-aware (hybrid) recommenders by incorporating metadata such as movie genres.
