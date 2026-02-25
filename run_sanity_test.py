#!/usr/bin/env python3
"""
Minimal sanity test – thin wrapper around run_experiments.py --sanity.

Injects --sanity before argument parsing so all sanity-test defaults
(single small model, tp=2, 8-token I/O, 4 prompts) are applied unless
explicitly overridden on the command line.

Examples:
    ./run_sanity_test.py                        # use all sanity defaults
    ./run_sanity_test.py --tp 4                 # sanity defaults, but tp=4
    ./run_sanity_test.py --models openai/gpt-oss-20b   # different model
"""

import sys

# Inject --sanity unless the user already passed it
if '--sanity' not in sys.argv:
    sys.argv.insert(1, '--sanity')

from run_experiments import main

sys.exit(main())
