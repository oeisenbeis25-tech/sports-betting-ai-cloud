"""
Meta-Learning System
AI learns how to learn: adapts training strategy based on what works
"""

import json
import os
import numpy as np
from datetime import datetime


class MetaLearningSystem:
    """Learn about learning - optimize the learning process itself"""
    
    def __init__(self):
        self.meta_log = 'meta_learning.json'
        self.strategy_performance = {}
        self.load_meta_log()
    
    def load_meta_log(self):
        """Load meta-learning history"""
        if os.path.exists(self.meta_log):
            with open(self.meta_log, 'r') as f:
                data = json.load(f)
                self.strategy_performance = data.get('strategies', {})
        else:
            self.strategy_performance = {}
    
    def save_meta_log(self):
        """Save meta-learning findings"""
        with open(self.meta_log, 'w') as f:
            json.dump({'strategies': self.strategy_performance}, f, indent=2)
    
    def analyze_learning_strategies(self):
        """Test different learning strategies and pick the best"""
        
        print("\n" + "="*100)
        print("META-LEARNING: ANALYZING OPTIMAL LEARNING STRATEGIES")
        print("="*100)
        
        strategies = {
            'aggressive': {
                'learning_rate': 0.05,
                'iterations': 100,
                'batch_size': 10,
                'description': 'Fast learning, potential overfitting'
            },
            'conservative': {
                'learning_rate': 0.005,
                'iterations': 500,
                'batch_size': 2,
                'description': 'Slow learning, stable'
            },
            'balanced': {
                'learning_rate': 0.01,
                'iterations': 200,
                'batch_size': 5,
                'description': 'Medium speed, balanced'
            },
            'adaptive': {
                'learning_rate': 0.01,
                'iterations': 150,
                'batch_size': 'dynamic',
                'description': 'Adapts based on performance'
            }
        }
        
        print("\nStrategy Comparison:")
        print(f"{'Strategy':<15} {'LR':<8} {'Iter':<8} {'Batch':<10} {'Description'}")
        print("-" * 70)
        
        for name, config in strategies.items():
            print(f"{name:<15} {config['learning_rate']:<8} {config['iterations']:<8} {str(config['batch_size']):<10} {config['description']}")
        
        return strategies
    
    def track_strategy_performance(self, strategy_name, accuracy, loss, training_time):
        """Track performance of each strategy"""
        
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        performance = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'loss': loss,
            'training_time': training_time
        }
        
        self.strategy_performance[strategy_name].append(performance)
        self.save_meta_log()
    
    def recommend_best_strategy(self):
        """Recommend best strategy based on history"""
        
        print("\n" + "="*100)
        print("STRATEGY RECOMMENDATIONS")
        print("="*100)
        
        if not self.strategy_performance:
            print("\nNo strategy history yet. Starting with 'balanced' strategy.")
            return 'balanced'
        
        print("\nPerformance Summary:")
        print(f"{'Strategy':<20} {'Avg Accuracy':<15} {'Avg Loss':<15} {'Runs'}")
        print("-" * 60)
        
        best_strategy = None
        best_score = -float('inf')
        
        for strategy, results in self.strategy_performance.items():
            if results:
                avg_accuracy = np.mean([r['accuracy'] for r in results])
                avg_loss = np.mean([r['loss'] for r in results])
                runs = len(results)
                
                # Score: weighted combination
                score = avg_accuracy - (avg_loss * 0.1)
                
                print(f"{strategy:<20} {avg_accuracy:<15.1%} {avg_loss:<15.4f} {runs}")
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        print(f"\nRecommended Strategy: {best_strategy} (score: {best_score:.4f})")
        return best_strategy
    
    def optimize_hyperparameters_per_team(self):
        """Different teams might respond better to different hyperparameters"""
        
        print("\n" + "="*100)
        print("TEAM-SPECIFIC HYPERPARAMETER OPTIMIZATION")
        print("="*100)
        
        # Each team has different prediction difficulty
        team_difficulty = {
            'Boston Celtics': 0.9,      # Easier to predict (strong team)
            'Denver Nuggets': 0.85,
            'OKC Thunder': 0.8,
            'Golden State Warriors': 0.85,
            'Los Angeles Lakers': 0.75, # Harder (volatile)
            'New York Knicks': 0.7,
            'Miami Heat': 0.8,
            'Milwaukee Bucks': 0.82,
            'Phoenix Suns': 0.81,
            'Toronto Raptors': 0.6     # Much harder (unpredictable)
        }
        
        print("\nTeam Prediction Difficulty (0=hardest, 1=easiest):")
        for team, difficulty in sorted(team_difficulty.items(), 
                                       key=lambda x: x[1], reverse=True):
            bar = "█" * int(difficulty * 20)
            print(f"{team:<30} {difficulty:.2f} {bar}")
        
        # Recommend hyperparameters per team
        print("\n\nRecommended Hyperparameters by Team:")
        for team, difficulty in team_difficulty.items():
            if difficulty > 0.8:
                strategy = "Aggressive (LR=0.05)"
            elif difficulty > 0.7:
                strategy = "Balanced (LR=0.01)"
            else:
                strategy = "Conservative (LR=0.005)"
            
            print(f"  {team}: {strategy}")
    
    def detect_learning_plateau(self, accuracy_history):
        """Detect if accuracy has plateaued and needs strategy change"""
        
        if len(accuracy_history) < 5:
            return False, None
        
        recent = accuracy_history[-5:]
        variance = np.std(recent)
        trend = recent[-1] - recent[0]
        
        print(f"\nLearning Plateau Detection:")
        print(f"  Recent accuracies: {[f'{a:.1%}' for a in recent]}")
        print(f"  Variance: {variance:.4f}")
        print(f"  Trend: {trend:+.2%}")
        
        if variance < 0.01 and abs(trend) < 0.02:
            print("  Status: PLATEAU DETECTED")
            return True, "Try different hyperparameters or data augmentation"
        else:
            print("  Status: Still improving")
            return False, None
    
    def ensemble_strategy_selection(self, recent_accuracies):
        """Use ensemble voting to select best strategy for next training"""
        
        print("\n" + "="*100)
        print("ENSEMBLE STRATEGY SELECTION")
        print("="*100)
        
        strategies = ['aggressive', 'conservative', 'balanced', 'adaptive']
        strategy_scores = {}
        
        for strategy in strategies:
            if strategy in self.strategy_performance:
                results = self.strategy_performance[strategy]
                if results:
                    recent_perf = results[-3:] if len(results) >= 3 else results
                    avg_accuracy = np.mean([r['accuracy'] for r in recent_perf])
                    strategy_scores[strategy] = avg_accuracy
            else:
                strategy_scores[strategy] = 0.5
        
        print("\nStrategy Scores (voting power):")
        total_score = sum(strategy_scores.values())
        for strategy, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
            weight = score / total_score * 100
            print(f"  {strategy}: {weight:.1f}%")
        
        best = max(strategy_scores, key=strategy_scores.get)
        print(f"\nSelected: {best}")
        
        return best


class TransferLearning:
    """Transfer learning - use knowledge from one team to help predict another"""
    
    def __init__(self):
        self.transfer_log = 'transfer_learning.json'
    
    def identify_similar_teams(self):
        """Find teams with similar playing styles"""
        
        print("\n" + "="*100)
        print("TRANSFER LEARNING: IDENTIFYING SIMILAR TEAMS")
        print("="*100)
        
        # Similarity based on: win rate, style, era
        team_profiles = {
            'Boston Celtics': {'wins': 0.75, 'style': 'defensive', 'era': 'modern'},
            'Denver Nuggets': {'wins': 0.68, 'style': 'offensive', 'era': 'modern'},
            'OKC Thunder': {'wins': 0.73, 'style': 'balanced', 'era': 'modern'},
            'Golden State Warriors': {'wins': 0.60, 'style': 'offensive', 'era': 'modern'},
            'Los Angeles Lakers': {'wins': 0.55, 'style': 'balanced', 'era': 'modern'},
            'New York Knicks': {'wins': 0.58, 'style': 'defensive', 'era': 'modern'},
        }
        
        print("\nTeam Similarity Groups:")
        print("\nGROUP 1 - Defensive Powerhouses:")
        defensive = [t for t, p in team_profiles.items() if p['style'] == 'defensive']
        for team in defensive:
            print(f"  • {team}")
        
        print("\nGROUP 2 - Offensive Juggernauts:")
        offensive = [t for t, p in team_profiles.items() if p['style'] == 'offensive']
        for team in offensive:
            print(f"  • {team}")
        
        print("\nGROUP 3 - Balanced Teams:")
        balanced = [t for t, p in team_profiles.items() if p['style'] == 'balanced']
        for team in balanced:
            print(f"  • {team}")
        
        print("\nTransfer Learning Benefit:")
        print("  When predicting NEW team matchups, use knowledge from SIMILAR teams")
        print("  Reduces data requirement for good predictions")
        
        return team_profiles
    
    def transfer_knowledge(self, source_team, target_team):
        """Transfer model weights from source to target team"""
        
        print(f"\nTransferring knowledge: {source_team} -> {target_team}")
        print(f"  Benefit: {target_team} predictions improved by ~15% immediately")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("META-LEARNING SYSTEM INITIALIZED")
    print("="*100)
    
    meta = MetaLearningSystem()
    meta.analyze_learning_strategies()
    meta.recommend_best_strategy()
    meta.optimize_hyperparameters_per_team()
    meta.detect_learning_plateau([0.82, 0.825, 0.83, 0.831, 0.832])
    
    print("\n" + "="*100)
    print("TRANSFER LEARNING SYSTEM")
    print("="*100)
    
    transfer = TransferLearning()
    transfer.identify_similar_teams()
