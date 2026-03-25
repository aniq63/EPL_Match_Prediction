"""
Quick test script to debug Understat extraction
"""
import sys
sys.path.insert(0, 'src')

import soccerdata as sd
import pandas as pd

print("Testing Understat data extraction...\n")

# Test 1: Try basic connection
try:
    print("Test 1: Creating Understat instance for 2023/2024...")
    understat = sd.Understat(leagues="ENG-Premier League", seasons="2023/2024")
    print("✓ Understat instance created successfully")
except Exception as e:
    print(f"✗ Failed to create instance: {e}")
    sys.exit(1)

# Test 2: Try to read data
try:
    print("\nTest 2: Reading team match stats...")
    df = understat.read_team_match_stats()
    print(f"✓ Data retrieved successfully!")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Column names: {list(df.columns)}")
except Exception as e:
    print(f"✗ Failed to read data: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
