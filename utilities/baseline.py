import matplotlib.pyplot as plt
import numpy as np
from rng_connector import Connector

def collect_bits(connector, source="rng1", total_bits=10000):
    bits = []
    connector.set_mode("MODE_UNWHITENED")
    while len(bits) < total_bits:
        ascii_data = connector.read(256, source)
        stream = connector.extract_lsb_stream_from_ascii(ascii_data)
        bits.extend(stream)
    return bits[:total_bits]

def plot_entropy_analysis(bits1, bits2, window=200):
    def rolling_mean(bits, window):
        return np.convolve(bits, np.ones(window)/window, mode='valid')

    def cumulative_deviation(bits):
        expected = 0.5
        cum_sum = np.cumsum(bits)
        expected_sum = np.arange(1, len(bits)+1) * expected
        return cum_sum - expected_sum

    x = np.arange(len(bits1))

    plt.figure(figsize=(14, 6))

    # Plot rolling mean
    plt.subplot(1, 2, 1)
    plt.plot(rolling_mean(bits1, window), label="RNG1", color='blue')
    plt.plot(rolling_mean(bits2, window), label="RNG2", color='orange')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title(f"Rolling Mean of LSBs (window={window})")
    plt.xlabel("Sample Index")
    plt.ylabel("Rolling Mean")
    plt.legend()

    # Plot cumulative deviation
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_deviation(bits1), label="RNG1", color='blue')
    plt.plot(cumulative_deviation(bits2), label="RNG2", color='orange')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Cumulative Deviation from 50%")
    plt.xlabel("Sample Index")
    plt.ylabel("Deviation")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    c = Connector()
    print("Collecting 10,000 LSBs from RNG1...")
    bits1 = collect_bits(c, source="rng1")
    print("Collecting 10,000 LSBs from RNG2...")
    bits2 = collect_bits(c, source="rng2")

    ones1 = sum(bits1)
    ones2 = sum(bits2)

    print(f"RNG1: {ones1} ones / 10,000 bits = {ones1 / 100:.2f}%")
    print(f"RNG2: {ones2} ones / 10,000 bits = {ones2 / 100:.2f}%")

    plot_entropy_analysis(bits1, bits2)
