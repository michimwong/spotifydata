#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:00:49 2024

@author: michiwong
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score
from sklearn.decomposition import PCA 
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import kstest
from scipy.stats import bootstrap 
from math import sqrt
import random 
import pandas as pd
from scipy.stats import norm

np.random.seed(15898164)

data = pd.read_csv('spotify52kData.csv')
data = np.vstack([data.columns, data.values])

#Isolates the song features and creates a subplot histogram of each feature, 
    #fitting a normal distribution to the feature and 
    #superimposing that normal distribution over the histogram.
feature_indices = [5, 7, 8, 10, 12, 13, 14, 15, 16, 17]

num_rows = 2
num_cols = 5
num_bins = 30

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 6))

for i, data_index in enumerate(feature_indices):
    row = i // num_cols
    col = i % num_cols
    
    feature_data = data[1:, data_index].astype(float)
    
    axes[row, col].hist(feature_data, bins=num_bins, density=True, alpha=0.7, color = "hotpink")
    axes[row, col].set_title(data[0, data_index].capitalize())
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')

    mu, std = norm.fit(feature_data)
    
    xmin, xmax = axes[row, col].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[row, col].plot(x, p, 'k', linewidth=2)

fig.tight_layout()
plt.show()

#Relationship between song length and popularity of a song
durationData = data[1:, 5].astype(float)
popularityData = data[1:, 4].astype(float)

valid_indices = ~np.isnan(durationData) & ~np.isnan(popularityData)
durationData = durationData[valid_indices]
popularityData = popularityData[valid_indices]

r = np.corrcoef(durationData,popularityData)

plt.scatter(durationData, popularityData, color = "hotpink", edgecolors='black') 
plt.xlabel('Song Length (ms)') 
plt.ylabel('Song Popularity')
plt.title('r = {:.3f}'.format(r[0,1])) 

#Popularity of explicit vs. not explicit songs
popularity = data[1:, 4].astype(int)
explicit = data[1:, 6].astype(bool)

explicit_group = popularity[explicit]
non_explicit_group = popularity[~explicit]

statistic, p_value = mannwhitneyu(explicit_group, non_explicit_group)


plt.figure(figsize=(8, 6))
plt.hist(non_explicit_group, bins=50, alpha=.65, label=f'Non-Explicit: Median={np.median(non_explicit_group):.2f}', color='cornflowerblue')
plt.hist(explicit_group, bins=50, alpha=.9, label=f'Explicit: Median={np.median(explicit_group):.2f}', color='hotpink')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Non-Explicit vs. Explicit Song Popularity')
plt.legend()
plt.grid(True)

plt.text(0.05, 0.05, f'p-value: {p_value:.2e}', fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

#Popularity of major key vs. nminor key songs

popularity = data[1:, 4].astype(int)
major = data[1:, 11].astype(bool)

major_group = popularity[major]
minor_group = popularity[~major]

statistic, p_value = mannwhitneyu(major_group, minor_group)


plt.figure(figsize=(8, 6))
plt.hist(major_group, bins=50, alpha=.65, label=f'Major: Median={np.median(major_group):.2f}', color='cornflowerblue')
plt.hist(minor_group, bins=50, alpha=.9, label=f'Minor: Median={np.median(minor_group):.2f}', color='hotpink')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Major vs. Minor Key Song Popularity')
plt.legend()
plt.grid(True)

plt.text(0.05, 0.05, f'p-value: {p_value:.2e}', fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

# Reflection of energy as the "loudness" of a song
energyData = data[1:, 8].astype(float)
loudnessData = data[1:, 10].astype(float)

valid_indices = ~np.isnan(energyData) & ~np.isnan(loudnessData)
energyData = energyData[valid_indices]
loudnessData = loudnessData[valid_indices]

r = np.corrcoef(energyData,loudnessData)

plt.scatter(energyData, loudnessData, color = "hotpink", edgecolors='black') 
plt.xlabel('Energy') 
plt.ylabel('Loudness (dB)')
plt.title('r = {:.3f}'.format(r[0,1])) 

#Which of the 10 individual (single) song features best predicts popularity?
feature_names = data[0:1, feature_indices]
features = data[1:, feature_indices]
popularity = data[1:, 4]   

X_train, X_test, y_train, y_test = train_test_split(features, popularity, test_size=0.2, random_state=15898164)

rmse_scores = {}
r2_scores = {}

for i in range(features.shape[1]):
    X_feature = X_train[:, i].reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X_feature, y_train)
    
    y_pred = model.predict(X_test[:, i].reshape(-1, 1))
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    rmse_scores[i] = rmse
    r2_scores[i] = r2

best_feature_index_rmse = min(rmse_scores, key=rmse_scores.get)
best_feature_index_r2 = max(r2_scores, key=r2_scores.get)

print("Best feature predicting popularity (RMSE):", feature_names[0, best_feature_index_rmse])
print("RMSE Score:", rmse_scores[best_feature_index_rmse])
print("Best feature predicting popularity (R2):", feature_names[0, best_feature_index_r2])
print("R2 Score:", r2_scores[best_feature_index_r2])

best_feature = X_train[:, best_feature_index_rmse].reshape(-1, 1)
best_model = LinearRegression()
best_model.fit(best_feature, y_train)

y_pred_best = best_model.predict(X_test[:, best_feature_index_rmse].reshape(-1, 1))
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, color='hotpink', edgecolors='black')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black')  # diagonal line
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs. Predicted Popularity (Best Linear Regression Model Using Instrumentalness)')
plt.text(0.8, 0.05, f'RMSE: {rmse_best:.2f}', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.8, 0.10, f'R^2: {r2_best:.2f}', fontsize=12, transform=plt.gca().transAxes)
plt.show()

#Building a model using *all* of the song features to predict popularity and comparing models
features = data[1:, feature_indices]
popularity = data[1:, 4]  

X_train, X_test, y_train, y_test = train_test_split(features, popularity, test_size=0.2, random_state=15898164)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

plt.scatter(y_test, y_pred, color='hotpink', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Plotting the diagonal line
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs. Predicted Popularity (Linear Regression Model Using All Features)')
plt.text(0.8, 0.05, f'RMSE: {rmse:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.text(0.8, 0.10, f'R^2: {r2:.2f}', fontsize=10, transform=plt.gca().transAxes)
plt.show()

#PCA
features = data[1:, feature_indices].astype(float)

zscoredData = stats.zscore(features)
pca = PCA().fit(zscoredData) 
eigVals = pca.explained_variance_
loadings = pca.components_ 
rotatedData = pca.fit_transform(zscoredData)

numQuestions = 10
x = np.linspace(1,numQuestions,numQuestions)
plt.title("Song Feature Data")
plt.bar(x, eigVals, color='hotpink')
plt.plot([0,numQuestions],[1,1],color='black') 
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

n_components = 3
explained_variance_ratio = pca.explained_variance_ratio_

meaningful_components_variance_ratio = explained_variance_ratio[:n_components]
sum_variance_ratio_meaningful = np.sum(meaningful_components_variance_ratio)
print(meaningful_components_variance_ratio)
print(sum_variance_ratio_meaningful)

#Predicting whether a song is in major key or minor key from valence
X = data[1:, 16].reshape(-1, 1) 
y = data[1:, 11].reshape(-1, 1).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15898164)

model = LogisticRegression()
model.fit(X_train, y_train)


plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='hotpink', label='Major')
plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='cornflowerblue', label='Minor')
plt.scatter(X_test[y_test == 1], y_test[y_test == 1], color='hotpink', marker='x')
plt.scatter(X_test[y_test == 0], y_test[y_test == 0], color='cornflowerblue', marker='x')

x_values = np.linspace(0, 1, 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_values, color='black', linestyle='--', label='Logistic Regression')

plt.xlabel('Valence')
plt.ylabel('Probability of Song Key')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

y_pred_prob = model.predict_proba(X_test)[:, 1]
auroc_score = roc_auc_score(y_test, y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='hotpink', lw=2, label='ROC curve (AUROC = %0.2f)' % auroc_score)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression Predicting Song Key From Valence')
plt.legend(loc="lower right")
plt.show()

X = data[1:, 8].reshape(-1, 1) 
y = data[1:, 11].reshape(-1, 1).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15898164)

model = LogisticRegression()
model.fit(X_train, y_train)


plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='hotpink', label='Major')
plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='cornflowerblue', label='Minor')
plt.scatter(X_test[y_test == 1], y_test[y_test == 1], color='hotpink', marker='x')
plt.scatter(X_test[y_test == 0], y_test[y_test == 0], color='cornflowerblue', marker='x')

x_values = np.linspace(0, 1, 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
plt.plot(x_values, y_values, color='black', linestyle='--', label='Logistic Regression')

plt.xlabel('Energy')
plt.ylabel('Probability of Song Key')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

y_pred_prob = model.predict_proba(X_test)[:, 1]
auroc_score = roc_auc_score(y_test, y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='hotpink', lw=2, label='ROC curve (AUROC = %0.2f)' % auroc_score)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression Predicting Song Key From Energy')
plt.legend(loc="lower right")
plt.show()

#Principal components vs. duration of a song used to predict whether a song is classical or not
def classify_genre(genre_column):
   
    classical_genres = ['classical']
    genre_column_lower = [genre.lower() for genre in genre_column]
    
    return [1 if genre in classical_genres else 0 for genre in genre_column_lower]

classified = np.array(classify_genre(data[1:, 19]))
X = data[1:, 5].reshape(-1, 1) 
y = classified.reshape(-1, 1).astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15898164)

model = LogisticRegression()
model.fit(X_train, y_train)


plt.figure(figsize=(10, 6))

plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='hotpink', label='Classical', marker='o')
plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='cornflowerblue', label='Not Classical', marker='o')
plt.scatter(X_test[y_test == 1], y_test[y_test == 1], color='hotpink', marker='x')
plt.scatter(X_test[y_test == 0], y_test[y_test == 0], color='cornflowerblue', marker='x')

x_values = np.linspace(X.min(), X.max(), 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

plt.plot(x_values, y_values, color='black', linestyle='--', label='Logistic Regression')

plt.xlabel('Duration')
plt.ylabel('Probability of Classical Song')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

y_pred_prob = model.predict_proba(X_test)[:, 1]
auroc_score = roc_auc_score(y_test, y_pred_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='hotpink', lw=2, label='ROC curve (AUROC = %0.2f)' % auroc_score)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression Predicting Classical Songs from Duration')
plt.legend(loc="lower right")
plt.show()

features = data[1:, feature_indices].astype(float)

zscoredData = stats.zscore(features)
pca = PCA().fit(zscoredData) 
eigVals = pca.explained_variance_
loadings = pca.components_ 
rotatedData = pca.fit_transform(zscoredData)

for i in range(3):
    X = rotatedData[:,i].reshape(-1, 1)
    
    y = classified.reshape(-1, 1).astype(int).ravel()
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='hotpink', label='Classical', marker='o')
    plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='cornflowerblue', label='Not Classical', marker='o')
    plt.scatter(X_test[y_test == 1], y_test[y_test == 1], color='hotpink', marker='x')
    plt.scatter(X_test[y_test == 0], y_test[y_test == 0], color='cornflowerblue', marker='x')
    
    x_values = np.linspace(X.min(), X.max(), 100)
    y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
    
    plt.plot(x_values, y_values, color='black', linestyle='--', label='Logistic Regression')
    
    plt.xlabel('First PCA Feature')
    plt.ylabel('Probability of Classical Song')
    plt.title(f'Logistic Regression for PCA Feature #{i+1}')
    plt.legend()
    plt.show()

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auroc_score = roc_auc_score(y_test, y_pred_prob)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='hotpink', lw=2, label='ROC curve (AUROC = %0.2f)' % auroc_score)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for Logistic Regression Predicting Classical Songs from PCA Feature #{i+1}')
    plt.legend(loc="lower right")
    plt.show()

#Do songs about love (meaning that they contain the word ‘love’ in their song title) 
    #differ in popularity than songs not about love?
popularity = data[1:, 4].astype(int)

titles = data[1:, 3].astype(str)
love_mask = np.array(['love' in title.lower() for title in titles], dtype=bool)
no_love_mask = ~love_mask

songs_with_love = popularity[love_mask]
songs_without_love = popularity[no_love_mask]

statistic, p_value = mannwhitneyu(songs_with_love, songs_without_love)


plt.figure(figsize=(8, 6))
plt.hist(non_explicit_group, bins=50, alpha=.65, label=f'Non-"Love" Songs: Median={np.median(songs_without_love):.2f}', color='cornflowerblue')
plt.hist(explicit_group, bins=50, alpha=.9, label=f'"Love" Songs: Median={np.median(songs_with_love):.2f}', color='hotpink')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of "Love" Songs vs. Non-"Love" Songs Popularity')
plt.legend()
plt.grid(True)

plt.text(0.05, 0.05, f'p-value: {p_value:.2e}', fontsize=12, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

plt.show()





