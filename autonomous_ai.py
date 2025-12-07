"""
Autonomous AI System - Self-Training Loop
Runs continuously, learns from every game, improves predictions automatically
"""

import json
import os
import time
import subprocess
from datetime import datetime, timedelta
import numpy as np
import threading


class AutonomousAISystem:
    """Self-training AI that runs forever"""
    
    def __init__(self):
        self.model_file = 'model_checkpoint.json'
        self.training_data_file = 'real_nba_seed_data.json'
        self.predictions_file = 'ai_predictions.json'
        self.learning_log = 'ai_learning_log.json'
        self.running = True
        self.load_or_create_checkpoint()
    
    def load_or_create_checkpoint(self):
        """Load model state or create new"""
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'version': 1,
                'created': datetime.now().isoformat(),
                'last_trained': None,
                'training_count': 0,
                'accuracy': 0.0,
                'games_learned': 0
            }
        
        self.load_learning_log()
    
    def load_learning_log(self):
        """Track all learning events"""
        if os.path.exists(self.learning_log):
            with open(self.learning_log, 'r') as f:
                self.learning_events = json.load(f)
        else:
            self.learning_events = []
    
    def save_checkpoint(self):
        """Save model state"""
        with open(self.model_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def save_learning_log(self):
        """Save learning events"""
        with open(self.learning_log, 'w') as f:
            json.dump(self.learning_events, f, indent=2)
    
    def fetch_completed_games(self):
        """Get newly completed games from ESPN"""
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for completed games...")
        
        try:
            # Import the game fetcher
            from real_games import RealGamesPredictor
            
            predictor = RealGamesPredictor()
            games = predictor.fetch_nba_games()
            
            # Filter only final games
            completed = [g for g in games if g.get('status') == 'final']
            
            print(f"Found {len(completed)} newly completed games")
            return completed
        
        except Exception as e:
            print(f"Error fetching games: {e}")
            return []
    
    def train_on_new_data(self):
        """Retrain model on all available real games"""
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting retraining...")
        
        try:
            # Import unified system
            from unified_real_data_ai import UnifiedRealDataAI
            
            system = UnifiedRealDataAI()
            accuracy = system.train_on_real_data()
            
            # Update checkpoint
            self.checkpoint['last_trained'] = datetime.now().isoformat()
            self.checkpoint['training_count'] += 1
            self.checkpoint['accuracy'] = accuracy
            
            # Get new games count
            with open(self.training_data_file, 'r') as f:
                data = json.load(f)
                games = len(data.get('games', []))
                self.checkpoint['games_learned'] = games
            
            self.save_checkpoint()
            
            print(f"✓ Training complete - Accuracy: {accuracy:.1f}% on {games} games")
            
            # Log learning event
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'training_completed',
                'accuracy': accuracy,
                'games': games
            }
            self.learning_events.append(event)
            self.save_learning_log()
            
            return True
        
        except Exception as e:
            print(f"✗ Training error: {e}")
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'training_error',
                'error': str(e)
            }
            self.learning_events.append(event)
            self.save_learning_log()
            return False
    
    def generate_predictions(self):
        """Generate predictions for upcoming games"""
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating predictions...")
        
        try:
            from future_games_scanner import FutureGamesScanner
            
            scanner = FutureGamesScanner()
            future_games = scanner.fetch_future_games()
            analyses = scanner.scan_for_value(future_games)
            
            # Save predictions
            with open(self.predictions_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'predictions': analyses
                }, f, indent=2)
            
            print(f"✓ Generated predictions for {len(analyses)} games")
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'predictions_generated',
                'count': len(analyses)
            }
            self.learning_events.append(event)
            self.save_learning_log()
            
            return len(analyses)
        
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return 0
    
    def autonomous_learning_cycle(self):
        """Main autonomous cycle: fetch → train → predict → repeat"""
        
        print("\n" + "="*100)
        print("AUTONOMOUS AI SYSTEM STARTED")
        print("="*100)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("System will run continuously, training on new data automatically")
        print("="*100 + "\n")
        
        cycle = 0
        
        while self.running:
            cycle += 1
            print(f"\n{'='*100}")
            print(f"CYCLE {cycle}")
            print(f"{'='*100}")
            
            try:
                # Step 1: Check for completed games
                completed = self.fetch_completed_games()
                
                # Step 2: Train if new data available
                if self.checkpoint['training_count'] == 0 or cycle % 3 == 0:
                    # Train every 3 cycles or first time
                    self.train_on_new_data()
                
                # Step 3: Generate predictions for upcoming games
                pred_count = self.generate_predictions()
                
                # Step 4: Check for high-value opportunities
                self.identify_value_bets()
                
                # Show status
                self.show_status()
                
                # Wait before next cycle (check every 30 minutes)
                print(f"\n⏳ Next cycle in 30 minutes...")
                for i in range(30 * 60):  # 30 minutes
                    if not self.running:
                        break
                    time.sleep(1)
            
            except Exception as e:
                print(f"\n✗ Cycle error: {e}")
                print("Continuing next cycle...")
                time.sleep(60)
    
    def identify_value_bets(self):
        """Analyze predictions for high-value opportunities"""
        
        try:
            if not os.path.exists(self.predictions_file):
                return
            
            with open(self.predictions_file, 'r') as f:
                data = json.load(f)
            
            predictions = data.get('predictions', [])
            
            # Find high-value bets (5%+ edge, 65%+ confidence)
            high_value = [
                p for p in predictions 
                if p.get('value_percent', 0) >= 5 
                and p.get('confidence', 0) >= 65
            ]
            
            if high_value:
                print(f"\n🎯 IDENTIFIED {len(high_value)} HIGH-VALUE BETTING OPPORTUNITIES:")
                for bet in high_value[:3]:
                    print(f"   • {bet.get('matchup', 'Unknown')}")
                    print(f"     Value: +{bet.get('value_percent', 0):.1f}% | Confidence: {bet.get('confidence', 0):.1f}%")
        
        except Exception as e:
            print(f"Value analysis error: {e}")
    
    def show_status(self):
        """Display current system status"""
        
        print(f"\n{'─'*100}")
        print("SYSTEM STATUS")
        print(f"{'─'*100}")
        print(f"Training cycles completed: {self.checkpoint['training_count']}")
        print(f"Last trained: {self.checkpoint['last_trained']}")
        print(f"Model accuracy: {self.checkpoint['accuracy']:.1f}%")
        print(f"Games learned from: {self.checkpoint['games_learned']}")
        print(f"Total learning events: {len(self.learning_events)}")
        print(f"{'─'*100}\n")
    
    def stop(self):
        """Stop the autonomous system"""
        print("\n⚠️ Stopping autonomous system...")
        self.running = False


def run_autonomous_system():
    """Start the autonomous AI system"""
    
    system = AutonomousAISystem()
    
    try:
        system.autonomous_learning_cycle()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        system.stop()


if __name__ == "__main__":
    run_autonomous_system()
