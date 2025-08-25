#!/usr/bin/env python3
"""
Comprehensive debug script to show exactly how belief model evaluation works
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from game28.constants import SUITS, RANKS, CARD_VALUES, TOTAL_POINTS
from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
import torch

def debug_belief_evaluation():
    """Debug belief model evaluation step by step"""
    print("🔍 Comprehensive Belief Model Evaluation Debug")
    print("=" * 80)
    
    try:
        # Create a game state
        print("1. Creating Game28State...")
        game_state = Game28State()
        print(f"   Game state created with {len(game_state.hands[0])} cards per player")
        
        # Load the belief model
        print("\n2. Loading belief model...")
        model = SimpleAdvancedBeliefNetwork()
        model_path = "models/belief_model/advanced_belief_model_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print("   ✓ Model loaded successfully")
        else:
            print("   ✗ Model not found, using untrained model")
        
        # Make a prediction
        print("\n3. Making belief model prediction...")
        with torch.no_grad():
            predictions = model.predict_beliefs(game_state, 0)
        
        print("   Prediction structure:")
        print(f"     Type: {type(predictions)}")
        print(f"     Trump suit shape: {predictions.trump_suit.shape}")
        print(f"     Uncertainty shape: {predictions.uncertainty.shape}")
        print(f"     Opponent hands keys: {list(predictions.opponent_hands.keys())}")
        print(f"     Void suits keys: {list(predictions.void_suits.keys())}")
        
        # Extract trump prediction
        print("\n4. Extracting trump prediction...")
        trump_probs = predictions.trump_suit.cpu().numpy().flatten()
        trump_suit = SUITS[trump_probs.argmax()]
        trump_confidence = trump_probs.max()
        print(f"   Trump probabilities: {trump_probs}")
        print(f"   Predicted trump suit: {trump_suit}")
        print(f"   Trump confidence: {trump_confidence:.4f}")
        
        # Extract opponent hand predictions
        print("\n5. Extracting opponent hand predictions...")
        opponent_strengths = {}
        for opp_id in range(4):
            if opp_id != 0 and opp_id in predictions.opponent_hands:  # Skip self (player 0)
                opp_hand_probs = predictions.opponent_hands[opp_id].cpu().numpy().flatten()
                print(f"   Opponent {opp_id} hand probabilities shape: {opp_hand_probs.shape}")
                print(f"   First 10 values: {opp_hand_probs[:10]}")
                
                # Calculate expected points for this opponent
                expected_points = 0.0
                print(f"   Calculating expected points for opponent {opp_id}:")
                for i, prob in enumerate(opp_hand_probs):
                    suit_idx = i // 8
                    rank_idx = i % 8
                    if rank_idx < len(RANKS):
                        card_value = CARD_VALUES[RANKS[rank_idx]]
                        contribution = float(prob) * card_value
                        expected_points += contribution
                        if i < 10:  # Show first 10 calculations
                            print(f"     Card {i}: suit={SUITS[suit_idx]}, rank={RANKS[rank_idx]}, prob={prob:.4f}, value={card_value}, contribution={contribution:.4f}")
                
                normalized_strength = float(expected_points / TOTAL_POINTS)
                opponent_strengths[opp_id] = normalized_strength
                print(f"   Total expected points: {expected_points:.4f}")
                print(f"   Normalized strength: {normalized_strength:.4f}")
        
        # Extract void suit predictions
        print("\n6. Extracting void suit predictions...")
        opponent_voids = {}
        for opp_id in range(4):
            if opp_id != 0 and opp_id in predictions.void_suits:  # Skip self (player 0)
                void_probs = predictions.void_suits[opp_id].cpu().numpy().flatten()
                print(f"   Opponent {opp_id} void probabilities: {void_probs}")
                
                void_suits = [SUITS[i] for i in range(4) if void_probs[i] > 0.5]
                opponent_voids[opp_id] = void_suits
                print(f"   Predicted void suits: {void_suits}")
        
        # Extract uncertainty
        print("\n7. Extracting uncertainty...")
        uncertainty = predictions.uncertainty.cpu().numpy().item()
        print(f"   Uncertainty: {uncertainty:.4f}")
        
        # Now simulate the evaluation process
        print("\n8. Simulating evaluation process...")
        
        # Test different scenarios
        scenarios = [
            ("Pass", -1),
            ("Bid 16", 16),
            ("Bid 20", 20),
            ("Bid 24", 24)
        ]
        
        for scenario_name, bid in scenarios:
            print(f"\n   --- {scenario_name} Evaluation ---")
            score = 0.0
            print(f"   Starting score: {score}")
            
            # 1. Trump prediction factor
            print(f"   1. Trump prediction factor (confidence: {trump_confidence:.4f}):")
            if trump_confidence > 0.8:  # High confidence in trump prediction
                # Check if we have trump cards
                our_trump_cards = sum(1 for card in game_state.hands[0] if card.suit == trump_suit)
                print(f"      Our trump cards: {our_trump_cards}")
                if our_trump_cards > 0:
                    trump_bonus = trump_confidence * our_trump_cards * 2.0
                    score += trump_bonus
                    print(f"      Bonus for having trump: +{trump_bonus:.4f}")
                else:
                    trump_penalty = trump_confidence * 1.0
                    score -= trump_penalty
                    print(f"      Penalty for not having trump: -{trump_penalty:.4f}")
            else:
                print(f"      Low confidence ({trump_confidence:.4f}), no trump factor")
            
            # 2. Opponent strength factor
            print(f"   2. Opponent strength factor:")
            avg_opponent_strength = sum(opponent_strengths.values()) / len(opponent_strengths) if opponent_strengths else 0.5
            print(f"      Average opponent strength: {avg_opponent_strength:.4f}")
            if avg_opponent_strength > 0.6:  # Strong opponents
                if bid > 0:
                    strength_penalty = avg_opponent_strength * 3.0
                    score -= strength_penalty
                    print(f"      Penalty for bidding against strong opponents: -{strength_penalty:.4f}")
                else:
                    strength_bonus = avg_opponent_strength * 1.0
                    score += strength_bonus
                    print(f"      Bonus for passing against strong opponents: +{strength_bonus:.4f}")
            else:  # Weak opponents
                if bid > 0:
                    strength_bonus = (1.0 - avg_opponent_strength) * 2.0
                    score += strength_bonus
                    print(f"      Bonus for bidding against weak opponents: +{strength_bonus:.4f}")
                else:
                    print(f"      No bonus/penalty for passing against weak opponents")
            
            # 3. Void suit factor
            print(f"   3. Void suit factor:")
            for opp_id, void_suits in opponent_voids.items():
                if isinstance(void_suits, (list, tuple)) and len(void_suits) > 0:
                    print(f"      Opponent {opp_id} void in: {void_suits}")
                    # Check if we have cards in those suits
                    for void_suit in void_suits:
                        our_suit_cards = sum(1 for card in game_state.hands[0] if card.suit == void_suit)
                        if our_suit_cards > 0:
                            void_bonus = our_suit_cards * 1.5
                            score += void_bonus
                            print(f"        Bonus for having {our_suit_cards} cards in {void_suit}: +{void_bonus:.4f}")
                        else:
                            print(f"        No cards in {void_suit}, no bonus")
                else:
                    print(f"      Opponent {opp_id}: no void suits predicted")
            
            # 4. Uncertainty factor
            print(f"   4. Uncertainty factor (uncertainty: {uncertainty:.4f}):")
            if uncertainty > 0.7:  # High uncertainty
                if bid > 0:
                    uncertainty_penalty = uncertainty * 2.0
                    score -= uncertainty_penalty
                    print(f"      Penalty for bidding when uncertain: -{uncertainty_penalty:.4f}")
                else:
                    uncertainty_bonus = uncertainty * 1.0
                    score += uncertainty_bonus
                    print(f"      Bonus for passing when uncertain: +{uncertainty_bonus:.4f}")
            else:
                print(f"      Low uncertainty ({uncertainty:.4f}), no uncertainty factor")
            
            # 5. Point prediction factor (simulated)
            print(f"   5. Point prediction factor (simulated):")
            point_prediction = 18.0  # Simulated value
            if bid > 0:
                if point_prediction >= bid:
                    point_bonus = 3.0
                    score += point_bonus
                    print(f"      Bonus for predicted points ({point_prediction}) >= bid ({bid}): +{point_bonus:.4f}")
                else:
                    point_penalty = 2.0
                    score -= point_penalty
                    print(f"      Penalty for predicted points ({point_prediction}) < bid ({bid}): -{point_penalty:.4f}")
            else:
                if point_prediction > 20:
                    point_penalty = 1.0
                    score -= point_penalty
                    print(f"      Penalty for high predicted points ({point_prediction}) when passing: -{point_penalty:.4f}")
                else:
                    print(f"      No penalty for low predicted points ({point_prediction}) when passing")
            
            print(f"   Final score for {scenario_name}: {score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_belief_evaluation()
