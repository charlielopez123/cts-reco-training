"""
Test script to verify imports work correctly before running the notebook.

Run this from the notebooks/ directory to verify your environment is set up correctly.
"""

import sys
from pathlib import Path

print("="*70)
print("TESTING IMPORT PATHS")
print("="*70)

# Setup paths
notebook_dir = Path(__file__).parent
project_root = notebook_dir.parent.parent

main_src = project_root / "src"
eval_src = project_root / "baselines_evaluation" / "src"

print(f"\n📁 Paths:")
print(f"  Notebook dir: {notebook_dir}")
print(f"  Project root: {project_root}")
print(f"  Main src: {main_src}")
print(f"  Eval src: {eval_src}")

print(f"\n✓ Checking paths exist:")
print(f"  Main src exists: {main_src.exists()}")
print(f"  Eval src exists: {eval_src.exists()}")
print(f"  Policies dir exists: {(eval_src / 'policies').exists()}")

# Add to path
sys.path.insert(0, str(main_src))
sys.path.insert(0, str(eval_src))

print(f"\n✓ sys.path updated:")
for i, p in enumerate(sys.path[:3]):
    print(f"  [{i}] {p}")

# Test imports
print(f"\n🔧 Testing imports...")

try:
    from cts_recommender.settings import get_settings
    print("  ✅ Main codebase import: OK")
except ImportError as e:
    print(f"  ❌ Main codebase import failed: {e}")

try:
    from policies.base import BasePolicy
    print("  ✅ Policies import: OK")
except ImportError as e:
    print(f"  ❌ Policies import failed: {e}")
    print(f"\n  Debug: Contents of eval_src/policies:")
    policies_dir = eval_src / 'policies'
    if policies_dir.exists():
        print(f"    {list(policies_dir.iterdir())}")

try:
    from policies import RandomPolicy, StaticScalarizationPolicy, AudienceOnlyPolicy, CTSPolicyAdapter
    print("  ✅ All policy classes import: OK")
except ImportError as e:
    print(f"  ❌ Policy classes import failed: {e}")

print(f"\n{'='*70}")
print("If all imports succeeded, the notebook should work!")
print("='*70")
