"""Fix sys.argv when running under Blender's --python mode.

When Blender runs:  blender --background --python script.py -- --arg1 val
sys.argv becomes:   ['blender', '--background', '--python', 'script.py', '--', '--arg1', 'val']

This module strips everything up to and including '--', leaving only the user args.
Import this BEFORE argparse.parse_args().
"""
import sys

def fix_argv_for_blender():
    """Strip Blender's CLI args, keeping only args after '--'."""
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        # Keep script name + args after '--'
        sys.argv = [sys.argv[0]] + sys.argv[idx + 1:]

# Auto-fix on import
fix_argv_for_blender()
