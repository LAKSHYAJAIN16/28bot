#!/usr/bin/env python3
"""
Fix opponent hand predictions flattening in main_game_simulation.py
"""

def fix_hand_probs():
    """Fix opponent hand predictions flattening"""
    print("🔧 Fixing opponent hand predictions flattening")
    print("=" * 50)
    
    try:
        # Read the file
        with open("main_game_simulation.py", "r") as f:
            content = f.read()
        
        # Replace all instances of opp_hand_probs extraction to flatten them
        old_pattern = "opp_hand_probs = belief_predictions.opponent_hands[opp_id].cpu().numpy()"
        new_pattern = "opp_hand_probs = belief_predictions.opponent_hands[opp_id].cpu().numpy().flatten()"
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"✓ Replaced opponent hand predictions with flattening")
        else:
            print("✗ No instances found to replace")
        
        # Write back to file
        with open("main_game_simulation.py", "w") as f:
            f.write(content)
        
        print("✅ Opponent hand predictions flattening fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    fix_hand_probs()
