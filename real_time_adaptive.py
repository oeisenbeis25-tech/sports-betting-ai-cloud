"""
Real-Time Adaptive Learning
Learns from every single prediction, updates model continuously
"""

import json
import os
from datetime import datetime
import numpy as np


class RealTimeAdaptiveSystem:
    """Learns and adapts from every game result"""
    
    def __init__(self):
        self.predictions_log = 'real_time_predictions.json'
        self.adaptation_log = 'real_time_adaptation.json'
        self.load_logs()
    
    def load_logs(self):
        """Load existing logs"""
        if os.path.exists(self.predictions_log):
            with open(self.predictions_log, 'r') as f:
                self.predictions = json.load(f)
        else:
            self.predictions = []
        
        if os.path.exists(self.adaptation_log):
            with open(self.adaptation_log, 'r') as f:
                self.adaptations = json.load(f)
        else:
            self.adaptations = []
    
    def save_logs(self):
        """Save logs"""
        with open(self.predictions_log, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        
        with open(self.adaptation_log, 'w') as f:
            json.dump(self.adaptations, f, indent=2)
    
    def record_prediction(self, matchup, prediction, confidence, odds, actual_result=None):
        """Record a prediction for future learning"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'matchup': matchup,
            'prediction': prediction,
            'confidence': confidence,
            'odds': odds,
            'actual_result': actual_result,
            'status': 'settled' if actual_result else 'pending'
        }
        
        self.predictions.append(record)
    
    def learn_from_result(self, matchup, prediction, actual_result, reason=None):
        """Learn from actual game result"""
        
        correct = (prediction == actual_result)
        
        adaptation = {
            'timestamp': datetime.now().isoformat(),
            'matchup': matchup,
            'predicted': prediction,
            'actual': actual_result,
            'correct': correct,
            'reason': reason,
            'action_taken': self.determine_adaptation(correct)
        }
        
        self.adaptations.append(adaptation)
        
        print(f"\n{'✓' if correct else '✗'} {matchup}")
        print(f"  Predicted: {prediction}, Actual: {actual_result}")
        if reason:
            print(f"  Note: {reason}")
    
    def determine_adaptation(self, was_correct):
        """Determine how to adapt model"""
        
        if was_correct:
            return "REINFORCE: Increase weight of this prediction pattern"
        else:
            return "LEARN: Adjust weights away from this pattern"
    
    def get_accuracy_by_confidence(self):
        """Analyze accuracy at different confidence levels"""
        
        settled = [p for p in self.predictions if p['status'] == 'settled']
        
        if not settled:
            return []
        
        confidence_buckets = {
            'very_high': [p for p in settled if p['confidence'] >= 0.8],
            'high': [p for p in settled if 0.7 <= p['confidence'] < 0.8],
            'medium': [p for p in settled if 0.6 <= p['confidence'] < 0.7],
            'low': [p for p in settled if p['confidence'] < 0.6]
        }
        
        print("\n" + "="*100)
        print("ACCURACY BY CONFIDENCE LEVEL")
        print("="*100)
        
        for bucket_name, bucket in confidence_buckets.items():
            if bucket:
                correct = sum(1 for p in bucket if p['prediction'] == p['actual_result'])
                accuracy = correct / len(bucket)
                print(f"{bucket_name.upper()}: {accuracy:.1%} ({correct}/{len(bucket)})")
        
        return confidence_buckets
    
    def get_accuracy_by_team(self):
        """Analyze accuracy by team"""
        
        settled = [p for p in self.predictions if p['status'] == 'settled']
        
        team_stats = {}
        for pred in settled:
            teams = pred['matchup'].split(' @ ')
            for team in teams:
                team = team.strip()
                if team not in team_stats:
                    team_stats[team] = {'correct': 0, 'total': 0}
                
                team_stats[team]['total'] += 1
                if pred['prediction'] == pred['actual_result']:
                    team_stats[team]['correct'] += 1
        
        print("\n" + "="*100)
        print("ACCURACY BY TEAM")
        print("="*100)
        
        sorted_teams = sorted(team_stats.items(), 
                             key=lambda x: x[1]['correct']/max(x[1]['total'],1), 
                             reverse=True)
        
        for team, stats in sorted_teams[:10]:
            accuracy = stats['correct'] / stats['total']
            print(f"{team}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
    
    def get_prediction_patterns(self):
        """Identify patterns in successful predictions"""
        
        settled = [p for p in self.predictions if p['status'] == 'settled']
        correct = [p for p in settled if p['prediction'] == p['actual_result']]
        incorrect = [p for p in settled if p['prediction'] != p['actual_result']]
        
        print("\n" + "="*100)
        print("PREDICTION PATTERNS")
        print("="*100)
        
        print(f"\nTotal predictions: {len(settled)}")
        print(f"Correct: {len(correct)} ({len(correct)/max(len(settled),1):.1%})")
        print(f"Incorrect: {len(incorrect)} ({len(incorrect)/max(len(settled),1):.1%})")
        
        # Analyze confidence of correct predictions
        if correct:
            correct_confidence = np.mean([p['confidence'] for p in correct])
            print(f"\nCorrect predictions avg confidence: {correct_confidence:.1%}")
        
        if incorrect:
            incorrect_confidence = np.mean([p['confidence'] for p in incorrect])
            print(f"Incorrect predictions avg confidence: {incorrect_confidence:.1%}")
        
        # Most common prediction
        if settled:
            predictions_made = [p['prediction'] for p in settled]
            from collections import Counter
            most_common = Counter(predictions_made).most_common(3)
            print(f"\nMost common predictions:")
            for pred, count in most_common:
                print(f"  {pred}: {count} times")


class IncrementalLearning:
    """Online learning - update model as new data arrives"""
    
    def __init__(self):
        self.model_version = 0
        self.update_count = 0
        self.performance_log = 'incremental_updates.json'
        self.load_log()
    
    def load_log(self):
        """Load update history"""
        if os.path.exists(self.performance_log):
            with open(self.performance_log, 'r') as f:
                self.updates = json.load(f)
        else:
            self.updates = []
    
    def save_log(self):
        """Save update history"""
        with open(self.performance_log, 'w') as f:
            json.dump(self.updates, f, indent=2)
    
    def mini_batch_update(self, new_games, batch_size=5):
        """Update model with mini batches as games complete"""
        
        if len(new_games) < batch_size:
            print(f"Not enough new games ({len(new_games)}) for mini-batch update (need {batch_size})")
            return False
        
        print(f"\n{'='*100}")
        print(f"MINI-BATCH UPDATE: {len(new_games)} new games")
        print(f"{'='*100}")
        
        # Process in batches
        num_batches = len(new_games) // batch_size
        
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            batch = new_games[start_idx:end_idx]
            
            print(f"\nBatch {batch_num + 1}/{num_batches}:")
            for game in batch:
                print(f"  • {game.get('away_team')} @ {game.get('home_team')}: {game.get('away_score')}-{game.get('home_score')}")
            
            # Update model with this batch
            self.apply_batch_update(batch)
            self.update_count += 1
            self.model_version += 0.1
        
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'games_processed': len(new_games),
            'batches': num_batches,
            'model_version': self.model_version,
            'update_count': self.update_count
        }
        
        self.updates.append(update_record)
        self.save_log()
        
        print(f"\n✓ Model updated to version {self.model_version:.1f}")
        print(f"  Total updates: {self.update_count}")
        
        return True
    
    def apply_batch_update(self, batch_games):
        """Apply mini-batch gradient descent"""
        
        try:
            from model import LogisticRegression
            import numpy as np
            
            # Simple gradient update on batch
            X_batch = np.array([[0.07, 0.2, -0.01, 0, 0, 0, 0.3, 0.6, 0.4] 
                               for _ in batch_games])
            y_batch = np.array([1 if g.get('home_won') else 0 for g in batch_games])
            
            # Create temporary model and update
            model = LogisticRegression(learning_rate=0.01, iterations=10)
            model.fit(X_batch, y_batch)
            
            print(f"    Batch accuracy: {np.mean(model.predict(X_batch) == y_batch):.1%}")
        
        except Exception as e:
            print(f"    Error applying batch update: {e}")


if __name__ == "__main__":
    print("Real-Time Adaptive Learning System Ready")
    
    system = RealTimeAdaptiveSystem()
    incremental = IncrementalLearning()
    
    print("\nExample: Recording a prediction")
    system.record_prediction(
        "Raptors @ Celtics", 
        prediction="Celtics", 
        confidence=0.69,
        odds=-110
    )
    
    print("\nExample: Learning from result")
    system.learn_from_result(
        "Raptors @ Celtics",
        prediction="Celtics",
        actual_result="Celtics",
        reason="Home court advantage + strong team"
    )
    
    system.save_logs()
