#!/usr/bin/env python3
"""
Fix all numpy array issues in main_game_simulation.py
"""

def fix_numpy_arrays():
    """Fix numpy array issues"""
    print("🔧 Fixing numpy array issues")
    print("=" * 50)
    
    try:
        # Read the file
        with open("main_game_simulation.py", "r") as f:
            content = f.read()
        
        # Replace all instances of prob * CARD_VALUES to ensure float conversion
        old_pattern = "expected_points += prob * CARD_VALUES[RANKS[rank_idx]]"
        new_pattern = "expected_points += float(prob) * CARD_VALUES[RANKS[rank_idx]]"
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"✓ Replaced prob * CARD_VALUES with float conversion")
        else:
            print("✗ No instances found to replace")
        
        # Replace all instances of expected_points / TOTAL_POINTS to ensure float conversion
        old_pattern2 = "opponent_strengths[opp_id] = expected_points / TOTAL_POINTS"
        new_pattern2 = "opponent_strengths[opp_id] = float(expected_points / TOTAL_POINTS)"
        
        if old_pattern2 in content:
            content = content.replace(old_pattern2, new_pattern2)
            print(f"✓ Replaced expected_points / TOTAL_POINTS with float conversion")
        else:
            print("✗ No instances found to replace")
        
        # Write back to file
        with open("main_game_simulation.py", "w") as f:
            f.write(content)
        
        print("✅ Numpy array issues fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    fix_numpy_arrays()
