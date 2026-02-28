import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


def main():
    with open("/home/fidel/Projects/corrqec2/data/experiment3_results.pkl", "rb") as f:
        data = pickle.load(f)
    print(data)


if __name__ == "__main__":
    main()
