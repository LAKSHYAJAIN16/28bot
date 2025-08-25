#!/usr/bin/env python3
"""
Debug script to show how card evaluation works with belief model
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from game28.constants import SUITS, RANKS, CARD_VALUES, TOTAL_POINTS, GamePhase
from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
import torch

def debug_card_evaluation():
    """Debug card evaluation step by step"""
    print("🎴 Comprehensive Card Evaluation Debug")
    print("=" * 80)
    
    try:
        # Create a game state
        print("1. Creating Game28State...")
        game_state = Game28State()
        print(f"   Game state created with {len(game_state.hands[0])} cards per player")
        
        # Set up a trick to make it more realistic
        print("\n2. Setting up a sample trick...")
        game_state.phase = GamePhase.CONCEALED
        game_state.trump_suit = "H"
        print(f"   Phase: {game_state.phase}")
        print(f"   Trump suit: {game_state.trump_suit}")
        
        # Load the belief model
        print("\n3. Loading belief model...")
        model = SimpleAdvancedBeliefNetwork()
        model_path = "models/belief_model/advanced_belief_model_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print("   ✓ Model loaded successfully")
        else:
            print("   ✗ Model not found, using untrained model")
        
        # Make a prediction
        print("\n4. Making belief model prediction...")
        with torch.no_grad():
            predictions = model.predict_beliefs(game_state, 0)
        
        # Extract all the predictions
        print("\n5. Extracting predictions...")
        trump_probs = predictions.trump_suit.cpu().numpy().flatten()
        trump_suit = SUITS[trump_probs.argmax()]
        trump_confidence = trump_probs.max()
        
        opponent_strengths = {}
        for opp_id in range(4):
            if opp_id != 0 and opp_id in predictions.opponent_hands:
                opp_hand_probs = predictions.opponent_hands[opp_id].cpu().numpy().flatten()
                expected_points = 0.0
                for i, prob in enumerate(opp_hand_probs):
                    suit_idx = i // 8
                    rank_idx = i % 8
                    if rank_idx < len(RANKS):
                        expected_points += float(prob) * CARD_VALUES[RANKS[rank_idx]]
                opponent_strengths[opp_id] = float(expected_points / TOTAL_POINTS)
        
        opponent_voids = {}
        for opp_id in range(4):
            if opp_id != 0 and opp_id in predictions.void_suits:
                void_probs = predictions.void_suits[opp_id].cpu().numpy().flatten()
                void_suits = [SUITS[i] for i in range(4) if void_probs[i] > 0.5]
                opponent_voids[opp_id] = void_suits
        
        uncertainty = predictions.uncertainty.cpu().numpy().item()
        
        print(f"   Trump prediction: {trump_suit} (confidence: {trump_confidence:.4f})")
        print(f"   Opponent strengths: {opponent_strengths}")
        print(f"   Opponent voids: {opponent_voids}")
        print(f"   Uncertainty: {uncertainty:.4f}")
        
        # Test different cards
        print("\n6. Testing card evaluation...")
        test_cards = game_state.hands[0][:3]  # Test first 3 cards
        
        for card in test_cards:
            print(f"\n   --- Evaluating Card: {card} ---")
            score = CARD_VALUES[card.rank]  # Base score
            print(f"   Base score from card value: {score}")
            
            # 1. Trump factor
            print(f"   1. Trump factor (confidence: {trump_confidence:.4f}):")
            if trump_confidence > 0.8 and card.suit == trump_suit:
                trump_multiplier = trump_confidence * 2.0
                score *= trump_multiplier
                print(f"      High confidence trump card! Score *= {trump_multiplier:.4f}")
            else:
                print(f"      Not a high-confidence trump card")
            
            # 2. Lead suit factor (simulate being in a trick)
            print(f"   2. Lead suit factor:")
            lead_suit = "D"  # Simulate diamonds led
            print(f"      Lead suit: {lead_suit}")
            if card.suit == lead_suit:
                # Check if we can win the trick
                current_high_card_value = 5  # Simulate current high card
                if CARD_VALUES[card.rank] > current_high_card_value:
                    lead_bonus = 5.0
                    score += lead_bonus
                    print(f"      Can win the trick! Bonus: +{lead_bonus:.4f}")
                else:
                    lead_bonus = 1.0
                    score += lead_bonus
                    print(f"      Following suit, small bonus: +{lead_bonus:.4f}")
            else:
                # Not following suit - check if we're void
                our_hand = game_state.hands[0]
                has_lead_suit = any(c.suit == lead_suit for c in our_hand)
                if not has_lead_suit:
                    # We're void - this is a trump or discard
                    if card.suit == trump_suit:
                        void_trump_bonus = 3.0
                        score += void_trump_bonus
                        print(f"      Void in {lead_suit}, trump bonus: +{void_trump_bonus:.4f}")
                    else:
                        void_discard_bonus = 0.5
                        score += void_discard_bonus
                        print(f"      Void in {lead_suit}, discard bonus: +{void_discard_bonus:.4f}")
                else:
                    void_penalty = 2.0
                    score -= void_penalty
                    print(f"      Not following suit when we could! Penalty: -{void_penalty:.4f}")
            
            # 3. Opponent void factor
            print(f"   3. Opponent void factor:")
            for opp_id, void_suits in opponent_voids.items():
                if isinstance(void_suits, (list, tuple)) and len(void_suits) > 0 and card.suit in void_suits:
                    void_bonus = 2.0
                    score += void_bonus
                    print(f"      Opponent {opp_id} void in {card.suit}! Bonus: +{void_bonus:.4f}")
                else:
                    print(f"      Opponent {opp_id}: no void bonus for {card.suit}")
            
            # 4. Opponent strength factor
            print(f"   4. Opponent strength factor:")
            avg_opponent_strength = sum(opponent_strengths.values()) / len(opponent_strengths) if opponent_strengths else 0.5
            print(f"      Average opponent strength: {avg_opponent_strength:.4f}")
            if avg_opponent_strength > 0.7:  # Strong opponents
                if card.rank in ['A', 'K', 'Q']:  # High cards
                    high_card_bonus = 3.0
                    score += high_card_bonus
                    print(f"      Strong opponents, high card bonus: +{high_card_bonus:.4f}")
                else:
                    print(f"      Strong opponents, but not a high card")
            else:  # Weak opponents
                if card.rank in ['7', '8', '9']:  # Low cards
                    low_card_bonus = 1.0
                    score += low_card_bonus
                    print(f"      Weak opponents, low card bonus: +{low_card_bonus:.4f}")
                else:
                    print(f"      Weak opponents, but not a low card")
            
            # 5. Uncertainty factor
            print(f"   5. Uncertainty factor (uncertainty: {uncertainty:.4f}):")
            if uncertainty > 0.8:  # High uncertainty
                if card.rank in ['A', 'K']:  # High cards
                    uncertainty_bonus = 2.0
                    score += uncertainty_bonus
                    print(f"      High uncertainty, high card bonus: +{uncertainty_bonus:.4f}")
                else:
                    print(f"      High uncertainty, but not a high card")
            else:
                print(f"      Low uncertainty, no uncertainty factor")
            
            # 6. Game phase factor
            print(f"   6. Game phase factor (phase: {game_state.phase}):")
            if game_state.phase == GamePhase.CONCEALED:
                # In concealed phase, be more conservative
                if card.rank in ['A', 'K']:
                    phase_bonus = 1.0
                    score += phase_bonus
                    print(f"      Concealed phase, high card bonus: +{phase_bonus:.4f}")
                else:
                    print(f"      Concealed phase, but not a high card")
            else:
                # In revealed phase, be more aggressive
                if card.suit == trump_suit:
                    phase_bonus = 2.0
                    score += phase_bonus
                    print(f"      Revealed phase, trump bonus: +{phase_bonus:.4f}")
                else:
                    print(f"      Revealed phase, but not trump")
            
            # 7. Trick position factor (simulate being last to play)
            print(f"   7. Trick position factor:")
            position = 3  # Simulate being last to play
            if position == 0:  # Leading
                if card.rank in ['A', 'K']:
                    lead_bonus = 2.0
                    score += lead_bonus
                    print(f"      Leading with high card bonus: +{lead_bonus:.4f}")
                else:
                    print(f"      Leading, but not a high card")
            elif position == 3:  # Last to play
                # Check if we can win
                current_winning_value = 8  # Simulate current winning card
                if CARD_VALUES[card.rank] > current_winning_value:
                    win_bonus = 4.0
                    score += win_bonus
                    print(f"      Can win the trick! Bonus: +{win_bonus:.4f}")
                else:
                    print(f"      Last to play, but can't win")
            else:
                print(f"      Middle position, no position factor")
            
            print(f"   Final score for {card}: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_card_evaluation()
