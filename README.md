# Py-Unfold examples

A few simple examples that show examples of unfolding on toy MC datasets.
Each script will pop up a few py-plot images, just close each one to get the next
one to appear.

`pyunfold_example.py` uses  fake MC data to unfold fake "detector" data.

`pyunfold_example_with_efficiencies.py` does the same basic unfolding but introduces
an non-uniform efficiency in between "truth" and "observed" dataset. This
is to demonstrate the cut efficiencies can be handled well.

`pyunfold_example_with_systematics.py` does the same unfolding but introduces a
systematic disagreement between "MC" and "data", but also introduces a systematic
variation into the unfolding procedure to account for that disagreement.
