"""
Feature selection and importance analysis for stock price prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import logging
from dataclasses import dataclass

from ..utils.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportance:
    """Feature importance results"""
    feature_names: List[str]
    importance_scores: List[float]
    method: str
    selected_features: List[str]

class FeatureSelector:
    """
    Advanced feature selection for stock price prediction
    """
    
    def __init__(self):
        """Initialize the feature selector"""
        self.importance_results = {}
        self.selected_features = []
        
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       method: str = "random_forest",
                       k_best: int = None,
                       threshold: float = None) -> List[str]:
        """
        Select best features using specified method
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Selection method ('random_forest', 'lasso', 'mutual_info', 'f_test', 'rfe')
            k_best: Number of best features to select
            threshold: Threshold for feature importance
            
        Returns:
            List of selected feature names
        """
        try:
            logger.info(f"Selecting features using {method}")
            
            if method == "random_forest":
                return self._select_with_random_forest(X, y, k_best, threshold)
            elif method == "lasso":
                return self._select_with_lasso(X, y, threshold)
            elif method == "mutual_info":
                return self._select_with_mutual_info(X, y, k_best)
            elif method == "f_test":
                return self._select_with_f_test(X, y, k_best)
            elif method == "rfe":
                return self._select_with_rfe(X, y, k_best)
            elif method == "combined":
                return self._select_with_combined_methods(X, y, k_best)
            else:
                raise ValueError(f"Unknown selection method: {method}")
                
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise PreprocessingError(f"Feature selection failed: {e}")
    
    def _select_with_random_forest(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series,
                                  k_best: int = None,
                                  threshold: float = None) -> List[str]:
        """Select features using Random Forest importance"""
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_names = X.columns.tolist()
        
        # Create importance results
        importance_result = FeatureImportance(
            feature_names=feature_names,
            importance_scores=importances.tolist(),
            method="random_forest",
            selected_features=[]
        )
        
        # Select features based on threshold or k_best
        if threshold is not None:
            selected_indices = np.where(importances >= threshold)[0]
            selected_features = [feature_names[i] for i in selected_indices]
        elif k_best is not None:
            selected_indices = np.argsort(importances)[-k_best:]
            selected_features = [feature_names[i] for i in selected_indices]
        else:
            # Use top 50% of features by default
            k_best = len(feature_names) // 2
            selected_indices = np.argsort(importances)[-k_best:]
            selected_features = [feature_names[i] for i in selected_indices]
        
        importance_result.selected_features = selected_features
        self.importance_results["random_forest"] = importance_result
        
        logger.info(f"Random Forest selected {len(selected_features)} features")
        return selected_features
    
    def _select_with_lasso(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          threshold: float = None) -> List[str]:
        """Select features using Lasso regularization"""
        
        # Use LassoCV for automatic alpha selection
        lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
        lasso.fit(X, y)
        
        # Get feature coefficients
        coefficients = np.abs(lasso.coef_)
        feature_names = X.columns.tolist()
        
        # Create importance results
        importance_result = FeatureImportance(
            feature_names=feature_names,
            importance_scores=coefficients.tolist(),
            method="lasso",
            selected_features=[]
        )
        
        # Select features with non-zero coefficients
        if threshold is not None:
            selected_indices = np.where(coefficients >= threshold)[0]
        else:
            selected_indices = np.where(coefficients > 0)[0]
        
        selected_features = [feature_names[i] for i in selected_indices]
        
        importance_result.selected_features = selected_features
        self.importance_results["lasso"] = importance_result
        
        logger.info(f"Lasso selected {len(selected_features)} features")
        return selected_features
    
    def _select_with_mutual_info(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               k_best: int = None) -> List[str]:
        """Select features using mutual information"""
        
        k_best = k_best or min(50, X.shape[1] // 2)
        
        # Calculate mutual information scores
        selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
        selector.fit(X, y)
        
        # Get selected features
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store importance results
        scores = selector.scores_
        importance_result = FeatureImportance(
            feature_names=X.columns.tolist(),
            importance_scores=scores.tolist(),
            method="mutual_info",
            selected_features=selected_features
        )
        self.importance_results["mutual_info"] = importance_result
        
        logger.info(f"Mutual information selected {len(selected_features)} features")
        return selected_features
    
    def _select_with_f_test(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          k_best: int = None) -> List[str]:
        """Select features using F-test"""
        
        k_best = k_best or min(50, X.shape[1] // 2)
        
        # F-test feature selection
        selector = SelectKBest(score_func=f_regression, k=k_best)
        selector.fit(X, y)
        
        # Get selected features
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store importance results
        scores = selector.scores_
        importance_result = FeatureImportance(
            feature_names=X.columns.tolist(),
            importance_scores=scores.tolist(),
            method="f_test",
            selected_features=selected_features
        )
        self.importance_results["f_test"] = importance_result
        
        logger.info(f"F-test selected {len(selected_features)} features")
        return selected_features
    
    def _select_with_rfe(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        k_best: int = None) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        
        k_best = k_best or min(50, X.shape[1] // 2)
        
        # Use Random Forest as base estimator for RFE
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=k_best)
        selector.fit(X, y)
        
        # Get selected features
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        # Store importance results
        rankings = selector.ranking_
        importance_result = FeatureImportance(
            feature_names=X.columns.tolist(),
            importance_scores=(1.0 / np.array(rankings)).tolist(),  # Convert ranking to importance
            method="rfe",
            selected_features=selected_features
        )
        self.importance_results["rfe"] = importance_result
        
        logger.info(f"RFE selected {len(selected_features)} features")
        return selected_features
    
    def _select_with_combined_methods(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    k_best: int = None) -> List[str]:
        """Select features using combination of methods"""
        
        methods = ["random_forest", "lasso", "mutual_info", "f_test"]
        feature_votes = {}
        
        # Get votes from each method
        for method in methods:
            try:
                selected_features = self.select_features(X, y, method=method, k_best=k_best)
                for feature in selected_features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        # Select features that were voted by at least 2 methods
        min_votes = max(1, len(methods) // 2)
        selected_features = [feature for feature, votes in feature_votes.items() 
                           if votes >= min_votes]
        
        # If too few features, take top voted features
        if k_best and len(selected_features) < k_best:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:k_best]]
        
        # Store combined results
        combined_result = FeatureImportance(
            feature_names=list(feature_votes.keys()),
            importance_scores=list(feature_votes.values()),
            method="combined",
            selected_features=selected_features
        )
        self.importance_results["combined"] = combined_result
        
        logger.info(f"Combined methods selected {len(selected_features)} features")
        return selected_features
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive feature importance report
        
        Returns:
            DataFrame with feature importance across different methods
        """
        if not self.importance_results:
            logger.warning("No importance results available")
            return pd.DataFrame()
        
        # Combine all results into a single DataFrame
        all_features = set()
        for result in self.importance_results.values():
            all_features.update(result.feature_names)
        
        all_features = sorted(list(all_features))
        
        # Create report DataFrame
        report_data = {"feature": all_features}
        
        for method, result in self.importance_results.items():
            importance_dict = dict(zip(result.feature_names, result.importance_scores))
            report_data[f"{method}_importance"] = [
                importance_dict.get(feature, 0) for feature in all_features
            ]
            report_data[f"{method}_selected"] = [
                feature in result.selected_features for feature in all_features
            ]
        
        report_df = pd.DataFrame(report_data)
        
        # Add summary statistics
        importance_columns = [col for col in report_df.columns if col.endswith("_importance")]
        selected_columns = [col for col in report_df.columns if col.endswith("_selected")]
        
        if importance_columns:
            report_df["avg_importance"] = report_df[importance_columns].mean(axis=1)
            report_df["max_importance"] = report_df[importance_columns].max(axis=1)
        
        if selected_columns:
            report_df["selection_count"] = report_df[selected_columns].sum(axis=1)
            report_df["selection_ratio"] = report_df["selection_count"] / len(selected_columns)
        
        # Sort by average importance
        if "avg_importance" in report_df.columns:
            report_df = report_df.sort_values("avg_importance", ascending=False)
        
        return report_df
    
    def analyze_feature_correlation(self, 
                                  X: pd.DataFrame,
                                  threshold: float = 0.95) -> Dict[str, List[str]]:
        """
        Analyze feature correlation and identify highly correlated features
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for identifying highly correlated features
            
        Returns:
            Dictionary of highly correlated feature groups
        """
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find highly correlated features
            highly_correlated = {}
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if correlation >= threshold:
                        group_key = f"{feature1}_{feature2}"
                        highly_correlated[group_key] = {
                            "features": [feature1, feature2],
                            "correlation": correlation
                        }
            
            logger.info(f"Found {len(highly_correlated)} highly correlated feature pairs")
            return highly_correlated
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return {}
    
    def remove_correlated_features(self, 
                                 X: pd.DataFrame,
                                 threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features, keeping the most important ones
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            List of features to keep
        """
        try:
            # Get correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find pairs of highly correlated features
            to_remove = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        
                        # Remove the feature with lower importance (if available)
                        if "random_forest" in self.importance_results:
                            rf_result = self.importance_results["random_forest"]
                            importance_dict = dict(zip(rf_result.feature_names, rf_result.importance_scores))
                            
                            importance1 = importance_dict.get(feature1, 0)
                            importance2 = importance_dict.get(feature2, 0)
                            
                            if importance1 < importance2:
                                to_remove.add(feature1)
                            else:
                                to_remove.add(feature2)
                        else:
                            # If no importance available, remove the second feature
                            to_remove.add(feature2)
            
            # Return features to keep
            features_to_keep = [col for col in X.columns if col not in to_remove]
            
            logger.info(f"Removed {len(to_remove)} correlated features, keeping {len(features_to_keep)}")
            return features_to_keep
            
        except Exception as e:
            logger.error(f"Error removing correlated features: {e}")
            return X.columns.tolist()
    
    def get_feature_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get comprehensive statistics for all features
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with feature statistics
        """
        try:
            stats = []
            
            for column in X.columns:
                series = X[column]
                
                stat_dict = {
                    "feature": column,
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "median": series.median(),
                    "skewness": series.skew(),
                    "kurtosis": series.kurtosis(),
                    "missing_count": series.isna().sum(),
                    "missing_ratio": series.isna().sum() / len(series),
                    "unique_count": series.nunique(),
                    "unique_ratio": series.nunique() / len(series)
                }
                
                stats.append(stat_dict)
            
            stats_df = pd.DataFrame(stats)
            return stats_df
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {e}")
            return pd.DataFrame()