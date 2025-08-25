#!/usr/bin/env python3
"""
Fix all void suit extraction issues in main_game_simulation.py
"""

def fix_void_suits():
    """Fix void suit extraction issues"""
    print("🔧 Fixing void suit extraction issues")
    print("=" * 50)
    
    try:
        # Read the file
        with open("main_game_simulation.py", "r") as f:
            content = f.read()
        
        # Replace all instances of void_probs extraction
        old_pattern = "void_probs = belief_predictions.void_suits[opp_id].cpu().numpy()"
        new_pattern = "void_probs = belief_predictions.void_suits[opp_id].cpu().numpy().flatten()"
        
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"✓ Replaced {content.count(new_pattern)} instances of void_probs extraction")
        else:
            print("✗ No instances found to replace")
        
        # Write back to file
        with open("main_game_simulation.py", "w") as f:
            f.write(content)
        
        print("✅ Void suit extraction issues fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        return False

if __name__ == "__main__":
    fix_void_suits()
