"""
Author: Josh Hellerstein
Date: 5/8/2019
File: util.py
"""

class Distribution(dict):
    def __missing__(self, key):
        # if missing, return 0
        return 0

    def renormalize(self):
        normalization_constant = sum(self.values())
        assert normalization_constant > 0, "Sum of probabilities is 0"
        for key in list(self.keys()):
            self[key] /= normalization_constant

def is_alpha(s):
	try:
		return s.encode('ascii').isalpha()
	except:
		return False