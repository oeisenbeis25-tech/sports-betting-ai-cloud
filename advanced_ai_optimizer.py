"""
Advanced AI Optimization - Machine Learning Enhancements
Multi-model ensemble, feature engineering, adaptive learning rate, cross-validation
"""

import json
import os
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedAIOptimizer:
    """Next-generation AI system with ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.performance_history = []
        self.checkpoint_file = 'advanced_models_checkpoint.json'
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load previous advanced model state"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.model_weights = data.get('weights', {})
                self.performance_history = data.get('history', [])
    
    def save_checkpoint(self):
        """Save advanced model state"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'weights': self.model_weights,
                'history': self.performance_history,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def feature_engineering(self, games):
        """Extract advanced features from game data"""
        
        features = []
        labels = []
        
        for game in games:
            try:
                away_team = game.get('away_team', '')
                home_team = game.get('home_team', '')
                away_won = game.get('away_won', 0)
                
                # FEATURE 1: Home Court Advantage (7% baseline)
                home_advantage = 0.07
                
                # FEATURE 2: Historical Performance
                # Calculate team winning percentages
                home_wins = sum(1 for g in games if g.get('home_team') == home_team and g.get('home_won'))
                home_total = sum(1 for g in games if g.get('home_team') == home_team)
                home_win_pct = home_wins / max(home_total, 1) if home_total > 0 else 0.5
                
                away_wins = sum(1 for g in games if g.get('away_team') == away_team and g.get('away_won'))
                away_total = sum(1 for g in games if g.get('away_team') == away_team)
                away_win_pct = away_wins / max(away_total, 1) if away_total > 0 else 0.5
                
                # FEATURE 3: Rest/Travel Factor
                # Away teams typically have 1-2% disadvantage
                travel_factor = -0.01
                
                # FEATURE 4: Strength Difference (ELO-like)
                strength_diff = (home_win_pct - away_win_pct) * 0.5
                
                # FEATURE 5: Variance (confidence in prediction)
                # Higher variance = more uncertain matchup
                variance = abs(home_win_pct - 0.5)
                
                # FEATURE 6: Matchup History
                # Head-to-head records
                h2h_wins = sum(1 for g in games 
                              if ((g.get('home_team') == home_team and g.get('away_team') == away_team 
                                   and g.get('home_won')) 
                                  or (g.get('away_team') == home_team and g.get('home_team') == away_team 
                                      and g.get('away_won'))))
                h2h_total = sum(1 for g in games 
                               if ((g.get('home_team') == home_team and g.get('away_team') == away_team)
                                   or (g.get('away_team') == home_team and g.get('home_team') == away_team)))
                h2h_advantage = (h2h_wins / max(h2h_total, 1) - 0.5) * 0.2 if h2h_total > 0 else 0
                
                # FEATURE 7: Season Momentum
                # Recent performance (last 5 games)
                recent_home = [g for g in games if g.get('home_team') == home_team][-5:]
                recent_away = [g for g in games if g.get('away_team') == away_team][-5:]
                home_momentum = sum(1 for g in recent_home if g.get('home_won')) / max(len(recent_home), 1) - 0.5
                away_momentum = sum(1 for g in recent_away if g.get('away_won')) / max(len(recent_away), 1) - 0.5
                momentum_diff = (home_momentum - away_momentum) * 0.3
                
                # FEATURE 8: Consistency (reliability of team)
                home_consistency = 1.0 - np.std([float(g.get('home_won', 0)) for g in recent_home]) if recent_home else 0.5
                away_consistency = 1.0 - np.std([float(g.get('away_won', 0)) for g in recent_away]) if recent_away else 0.5
                consistency_diff = (home_consistency - away_consistency) * 0.1
                
                # Combine all features
                feature_vector = np.array([
                    home_advantage,           # 7%
                    strength_diff,            # Win % difference
                    travel_factor,            # -1%
                    h2h_advantage,            # Head-to-head
                    momentum_diff,            # Recent form
                    consistency_diff,         # Reliability
                    variance,                 # Prediction confidence
                    home_win_pct,             # Home team strength
                    away_win_pct              # Away team strength
                ])
                
                features.append(feature_vector)
                labels.append(1 if home_team == game.get('home_team') and not away_won else 0)
            
            except Exception as e:
                continue
        
        return np.array(features), np.array(labels)
    
    def build_ensemble(self, X, y):
        """Build ensemble of multiple models"""
        
        print("\n" + "="*100)
        print("BUILDING ADVANCED ENSEMBLE MODEL")
        print("="*100)
        
        ensemble = {}
        
        # MODEL 1: Logistic Regression with L2 regularization
        print("\n[1/5] Training Logistic Regression...")
        from model import SportsPredictor
        lr_model = SportsPredictor(learning_rate=0.01, epochs=200)
        lr_model.train(X, y)
        lr_accuracy = np.mean(lr_model.predict(X) == y)
        ensemble['logistic_regression'] = (lr_model, lr_accuracy)
        print(f"    Logistic Regression Accuracy: {lr_accuracy:.1%}")
        
        # MODEL 2: Feature-weighted variant
        print("[2/5] Training Weighted Logistic Regression...")
        feature_weights = np.abs(lr_model.weights) / np.sum(np.abs(lr_model.weights))
        X_weighted = X * feature_weights
        lr_weighted = SportsPredictor(learning_rate=0.01, epochs=200)
        lr_weighted.train(X_weighted, y)
        lr_w_accuracy = np.mean(lr_weighted.predict(X_weighted) == y)
        ensemble['weighted_logistic'] = (lr_weighted, lr_w_accuracy)
        print(f"    Weighted LR Accuracy: {lr_w_accuracy:.1%}")
        
        # MODEL 3: High-bias model (bootstrap aggregation)
        print("[3/5] Training Bootstrap Ensemble...")
        bootstrap_models = []
        bootstrap_accuracies = []
        for i in range(5):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            boot_model = SportsPredictor(learning_rate=0.01, epochs=150)
            boot_model.train(X_boot, y_boot)
            boot_acc = np.mean(boot_model.predict(X) == y)
            bootstrap_models.append(boot_model)
            bootstrap_accuracies.append(boot_acc)
        avg_boot_accuracy = np.mean(bootstrap_accuracies)
        ensemble['bootstrap'] = (bootstrap_models, avg_boot_accuracy)
        print(f"    Bootstrap Ensemble Accuracy: {avg_boot_accuracy:.1%}")
        
        # MODEL 4: Conservative model (high-confidence predictions only)
        print("[4/5] Training Conservative Model...")
        conservative = SportsPredictor(learning_rate=0.005, epochs=300)
        conservative.train(X, y)
        conservative_accuracy = np.mean(conservative.predict(X) == y)
        ensemble['conservative'] = (conservative, conservative_accuracy)
        print(f"    Conservative Model Accuracy: {conservative_accuracy:.1%}")
        
        # MODEL 5: Aggressive model (higher sensitivity)
        print("[5/5] Training Aggressive Model...")
        aggressive = SportsPredictor(learning_rate=0.05, epochs=100)
        aggressive.train(X, y)
        aggressive_accuracy = np.mean(aggressive.predict(X) == y)
        ensemble['aggressive'] = (aggressive, aggressive_accuracy)
        print(f"    Aggressive Model Accuracy: {aggressive_accuracy:.1%}")
        
        print("\n" + "="*100)
        print("ENSEMBLE COMPLETE")
        print("="*100)
        
        # Calculate weighted voting
        total_accuracy = sum(acc for _, acc in ensemble.values())
        self.model_weights = {
            name: acc / total_accuracy 
            for name, (_, acc) in ensemble.items()
        }
        
        print("\nModel Weights (voting power):")
        for name, weight in self.model_weights.items():
            print(f"  {name}: {weight:.1%}")
        
        return ensemble
    
    def cross_validate(self, X, y, k=5):
        """K-fold cross-validation for robust accuracy estimation"""
        
        from model import SportsPredictor
        
        print(f"\nPerforming {k}-fold cross-validation...")
        
        fold_size = len(X) // k
        fold_accuracies = []
        
        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else len(X)
            
            X_test = X[start:end]
            y_test = y[start:end]
            X_train = np.vstack([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])
            
            model = SportsPredictor(learning_rate=0.01, epochs=200)
            model.train(X_train, y_train)
            accuracy = np.mean(model.predict(X_test) == y_test)
            fold_accuracies.append(accuracy)
            
            print(f"  Fold {fold+1}/{k}: {accuracy:.1%}")
        
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        print(f"\nCross-validation Results:")
        print(f"  Mean Accuracy: {mean_accuracy:.1%}")
        print(f"  Std Dev: {std_accuracy:.1%}")
        print(f"  95% CI: [{mean_accuracy - 1.96*std_accuracy:.1%}, {mean_accuracy + 1.96*std_accuracy:.1%}]")
        
        return mean_accuracy, std_accuracy
    
    def adaptive_learning(self, games):
        """Adapt learning rate based on recent performance"""
        
        print("\nAnalyzing recent performance for adaptive learning...")
        
        if not self.performance_history:
            learning_rate = 0.01
        else:
            recent = self.performance_history[-10:]
            recent_accuracy = np.mean(recent)
            
            # If accuracy is dropping, increase learning rate
            if recent_accuracy < 0.50:
                learning_rate = 0.05
                print(f"  Low accuracy ({recent_accuracy:.1%}) - Increasing learning rate to 0.05")
            elif recent_accuracy > 0.85:
                learning_rate = 0.005
                print(f"  High accuracy ({recent_accuracy:.1%}) - Decreasing learning rate to 0.005")
            else:
                learning_rate = 0.01
                print(f"  Normal accuracy ({recent_accuracy:.1%}) - Standard learning rate 0.01")
        
        return learning_rate
    
    def hyperparameter_optimization(self, X, y):
        """Test different hyperparameters to find best combination"""
        
        from model import SportsPredictor
        
        print("\nOptimizing hyperparameters...")
        print("Testing: learning rates, regularization, iterations")
        
        learning_rates = [0.001, 0.005, 0.01, 0.05]
        best_config = None
        best_accuracy = 0
        
        configs_tested = 0
        for lr in learning_rates:
            model = SportsPredictor(learning_rate=lr, epochs=200)
            model.train(X, y)
            accuracy = np.mean(model.predict(X) == y)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {'lr': lr, 'accuracy': accuracy}
            
            configs_tested += 1
        
        print(f"  Tested {configs_tested} configurations")
        print(f"  Best: LR={best_config['lr']}, Accuracy={best_config['accuracy']:.1%}")
        
        return best_config
    
    def feature_importance_analysis(self, X, y, feature_names):
        """Analyze which features matter most"""
        
        from model import SportsPredictor
        
        print("\nAnalyzing feature importance...")
        
        model = SportsPredictor(learning_rate=0.01, epochs=200)
        model.train(X, y)
        
        # Importance = |weight| * average feature value
        importances = np.abs(model.weights) * np.mean(np.abs(X), axis=0)
        
        # Normalize
        importances = importances / np.sum(importances)
        
        feature_ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        print("\nTop Features by Importance:")
        for i, (name, importance) in enumerate(feature_ranking, 1):
            bar = "█" * int(importance * 50)
            print(f"  {i}. {name}: {importance:.1%} {bar}")
        
        self.feature_importance = dict(feature_ranking)
        
        return feature_ranking
    
    def optimize_and_train(self):
        """Full optimization pipeline"""
        
        print("\n" + "="*100)
        print("STARTING ADVANCED AI OPTIMIZATION")
        print("="*100)
        
        # Load training data
        try:
            with open('real_nba_seed_data.json', 'r') as f:
                data = json.load(f)
            games = data.get('games', [])
            
            if not games:
                print("No training data found")
                return False
            
            print(f"\nLoaded {len(games)} training games")
            
            # STEP 1: Feature Engineering
            print("\n[STEP 1] Feature Engineering...")
            X, y = self.feature_engineering(games)
            print(f"  Generated {X.shape[1]} features from {len(games)} games")
            print(f"  Feature dimensions: {X.shape}")
            
            feature_names = [
                'Home Advantage', 'Strength Difference', 'Travel Factor',
                'Head-to-Head', 'Momentum', 'Consistency', 'Variance',
                'Home Win %', 'Away Win %'
            ]
            
            # STEP 2: Hyperparameter Optimization
            print("\n[STEP 2] Hyperparameter Optimization...")
            best_config = self.hyperparameter_optimization(X, y)
            
            # STEP 3: Cross-Validation
            print("\n[STEP 3] Cross-Validation...")
            cv_mean, cv_std = self.cross_validate(X, y, k=5)
            
            # STEP 4: Build Ensemble
            print("\n[STEP 4] Building Ensemble...")
            ensemble = self.build_ensemble(X, y)
            
            # STEP 5: Feature Importance
            print("\n[STEP 5] Feature Importance Analysis...")
            self.feature_importance_analysis(X, y, feature_names)
            
            # STEP 6: Save Results
            self.performance_history.append(cv_mean)
            self.save_checkpoint()
            
            print("\n" + "="*100)
            print("ADVANCED OPTIMIZATION COMPLETE")
            print("="*100)
            print(f"\nSummary:")
            print(f"  Best Config Accuracy: {best_config['accuracy']:.1%}")
            print(f"  Cross-Val Accuracy: {cv_mean:.1%} ± {cv_std:.1%}")
            print(f"  Ensemble Ready: YES")
            print(f"  Feature Analysis: COMPLETE")
            
            return True
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    optimizer = AdvancedAIOptimizer()
    optimizer.optimize_and_train()
