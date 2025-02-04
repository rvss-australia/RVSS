# Simple pickler
#
# See install_notes for dependencies
# 
# Feb 2025 - Created for / presented at RVSS 2025 by Don Dansereau
import pickle


def save_model(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {filename}")

def load_model(filename):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model loaded from {filename}")
    return params