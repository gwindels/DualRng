
import serial
import time
import os
import numpy as np
from serial.tools import list_ports

# Supported Modes
# MODE_NORMAL       300       /* Streams combined + Mersenne Twister */
# MODE_PSDEBUG      1200      /* PS Voltage in mV in ASCII */
# MODE_RNGDEBUG     2400      /* RNG Debug 0x0RRR 0x0RRR in ASCII */
# MODE_RNG1WHITE    4800      /* RNG1 + Mersenne Twister */
# MODE_RNG2WHITE    9600      /* RNG2 + Mersenns Twister*/
# MODE_RAW_BIN      19200     /* Raw ADC Samples in Binary Mode */
# MODE_RAW_ASC      38400     /* Raw ADC Samples in Ascii Mode */
# MODE_UNWHITENED  57600      /* Unwhitened RNG1-RNG2 (TrueRNGproV2 Only) */
# MODE_NORMAL_ASC   115200    /* Normal in Ascii Mode (TrueRNGproV2 Only) */
# MODE_NORMAL_ASC_SLOW 230400    /* Normal in Ascii Mode - Slow for small devices (TrueRNGproV2 Only) */

class Connector():
    def __init__(self, mode = False):
        self.mode_supported = True
        self.mode = mode
        self._ser = None           # Persistent serial connection (None = one-shot mode)
        self._current_rng = None   # Last RNG source sent, for mode tracking
        self.port = self.identify_rng_port()
        if mode:
            self.set_mode(self.mode)
    
    def identify_rng_port(self):
        rng_com_port = None
        for temp in list_ports.comports():
            if '04D8:F5FE' in temp[2]:
                # print('Found TrueRNG on ' + temp[0])
                if rng_com_port == None:        # always chooses the 1st TrueRNG found
                    rng_com_port=temp[0]
                    self.mode_supported = False
            if '16D0:0AA0' in temp[2]:
                # print('Found TrueRNGpro on ' + temp[0])
                if rng_com_port == None:        # always chooses the 1st TrueRNG found
                    rng_com_port=temp[0]
            if '04D8:EBB5' in temp[2]:
                # print('Found TrueRNGproV2 on ' + temp[0])
                if rng_com_port == None:        # always chooses the 1st TrueRNG found
                    rng_com_port=temp[0]
        # print(f"Using com port:  {str(rng_com_port)}")
        if rng_com_port is None:
            raise Exception("TrueRNG Device not found")
        return rng_com_port
    
    def set_mode(self, mode:str) -> bool:
        # Confirm mode is supported
        if not self.mode_supported:
            return False
        # Local variables
        PORT = self.port
        MODE = mode
        mode_match = 0
        # "Knock" Sequence to activate mode change
        ser = serial.Serial(port=PORT,baudrate=110,timeout=1)
        ser.close()
        ser = serial.Serial(port=PORT,baudrate=300,timeout=1)
        ser.close()
        ser = serial.Serial(port=PORT,baudrate=110,timeout=1)
        ser.close()
        # Set mode
        if MODE=='MODE_NORMAL':
            ser = serial.Serial(port=PORT,baudrate=300,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_PSDEBUG':
            ser = serial.Serial(port=PORT,baudrate=1200,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_RNGDEBUG':
            ser = serial.Serial(port=PORT,baudrate=2400,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_RNG1WHITE':
            ser = serial.Serial(port=PORT,baudrate=4800,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_RNG2WHITE':
            ser = serial.Serial(port=PORT,baudrate=9600,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_RAW_BIN':
            ser = serial.Serial(port=PORT,baudrate=19200,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_RAW_ASC':
            ser = serial.Serial(port=PORT,baudrate=38400,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_UNWHITENED':
            ser = serial.Serial(port=PORT,baudrate=57600,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_NORMAL_ASC':
            ser = serial.Serial(port=PORT,baudrate=115200,timeout=1)
            # print('Switched to ' + MODE)
            mode_match=1
        if MODE=='MODE_NORMAL_ASC_SLOW':
            ser = serial.Serial(port=PORT,baudrate=230400,timeout=1)
            mode_match=1
        ser.close()
        if mode_match==0:
            print('Mode not Recognized')
            return False
        else:
            return True
        
    def open(self):
        """Open a persistent serial connection with one-time soft reset.
        Idempotent — no-op if already open."""
        if self._ser is not None:
            return
        self._ser = serial.Serial(self.port, timeout=1)
        self._ser.write(b'\xB0')  # Soft reset
        time.sleep(0.1)
        self._ser.setDTR(True)
        self._ser.flushInput()
        self._current_rng = None  # Force first read to send mode command

    def close(self):
        """Close the persistent serial connection. Safe to call multiple times."""
        self._current_rng = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def bits_to_trials(self, data):
        # Convert raw byte data to 200-bit trial results
        if not data:
            return np.array([])
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        if len(bits) < 200:
            return np.array([])
        return np.array([np.sum(chunk) for chunk in np.array_split(bits, len(bits)//200)])
    
    def bits_to_binary_trial(self, data):
        # Return 1 if majority of bits are 1s, else 0. Uses odd number of bits
        if not data:
            return None

        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        n_bits = len(bits)

        if n_bits < 1:
            return None
        if n_bits % 2 == 0:
            n_bits -= 1
        bits = bits[:n_bits]

        ones = np.sum(bits)
        # print(f"Total bits: {n_bits}, 1s: {ones}, 0s: {n_bits - ones}")
        return int(ones > (n_bits // 2))

    def normalize_trial_count(self, trial_counts, window_size=100):
        """
        Normalize a sequence of trial counts using Z-score normalization.
        trial_counts: numpy array of trial sums from bits_to_trials.
        window_size: number of trials for rolling baseline.
        Returns the normalized Z-score of the last trial.
        """
        if len(trial_counts) < 2:
            return 0.0
        recent = trial_counts[-window_size:] if len(trial_counts) >= window_size else trial_counts
        mean = np.mean(recent)
        std = np.std(recent)
        if std == 0:
            return 0.0
        return (trial_counts[-1] - mean) / std

    def normalized_binary_trial(self, data, baseline=None):
        """
        Convert data to binary trial (0/1) and normalize against a baseline bias.
        baseline: expected probability of 1 (if None, defaults to 0.5).
        Returns: normalized result where 0.5 is baseline.
        """
        result = self.bits_to_binary_trial(data)
        if baseline is None:
            baseline = 0.5
        return result - baseline

    def normalized_trial_counts(self, data, baseline_mean=None):
        """
        Convert data to trial counts and center by baseline mean.
        baseline_mean: expected average number of 1s in a 200-bit chunk (default 100).
        Returns: centered count values.
        """
        counts = self.bits_to_trials(data)
        if baseline_mean is None:
            baseline_mean = 100  # expected mean of 1s in 200 bits
        return counts - baseline_mean

    def extract_lsb_stream_from_binary(self, raw_bytes):
        """
        Extract LSBs from each byte in raw binary unwhitened mode.
        Returns a list of 0s and 1s.
        """
        if not raw_bytes:
            return []
        return [byte & 1 for byte in raw_bytes]

    def extract_lsb_stream_from_ascii(self, raw_ascii):
        """
        Extracts LSBs from RAW ASCII mode data like b'257,300,177,...'
        Returns a list of bits (0 or 1).
        """
        try:
            byte_string = raw_ascii.decode('utf-8').strip(',\n ')
            adc_values = [int(x) for x in byte_string.split(',') if x.strip().isdigit()]
            return [val & 1 for val in adc_values]  # Extract LSB
        except:
            return []

    def compensate_lsb_bias(self, bits, expected_bias=0.5):
        """
        Center a list of bits (0 or 1) by subtracting expected LSB bias.
        Returns a centered float.
        """
        if not bits:
            return 0.0
        actual_mean = np.mean(bits)
        return actual_mean - expected_bias

    def lsb_based_trial(self, bits, chunk_size=200, expected_bias=0.5):
        """
        Use LSB bits to compute one trial result, compensated for known bias.
        Returns 1 if above expected threshold, else 0.
        """
        if len(bits) < chunk_size:
            return None
        chunk = bits[:chunk_size]
        ones = np.sum(chunk)
        return int(ones > (expected_bias * chunk_size))

    def lsb_z_score(self, bits, chunk_size=200, expected_bias=0.5):
        """
        Compute Z-score of 1s in an LSB chunk relative to expected biased mean.
        """
        if len(bits) < chunk_size:
            print(f"[Z-Score Skipped] Not enough bits: {len(bits)} < {chunk_size}")
            return 0.0
        chunk = bits[:chunk_size]
        mean = expected_bias * chunk_size
        std = np.sqrt(chunk_size * expected_bias * (1 - expected_bias))
        actual_z = (np.sum(chunk) - mean) / std if std > 0 else 0.0
        # print(f"[Z-Score] Ones: {np.sum(chunk)}, Mean: {mean}, Std: {std:.4f}, Z: {actual_z:.4f}")
        return actual_z

    def split_interleaved_rngs(self, raw_bytes):
        """
        Splits interleaved RNG1 and RNG2 bytes from MODE_UNWHITENED default mode.
        Returns (rng1_bytes, rng2_bytes)
        """
        rng1 = raw_bytes[::2]
        rng2 = raw_bytes[1::2]
        return rng1, rng2

        
    def read(self, block_size: int = 32, rng = False):
        if self._ser is not None:
            return self._read_persistent(block_size, rng)
        return self._read_oneshot(block_size, rng)

    def _read_oneshot(self, block_size, rng):
        """Original one-shot behavior: open, reset, read, close."""
        with serial.Serial(self.port, timeout=1) as ser:
            ser.write(b'\xB0')  # Soft reset
            time.sleep(0.1)
            ser.setDTR(True)
            ser.flushInput()

            # Set device mode
            if rng == "rng1":
                ser.write(b'\xC4')
            elif rng == "rng2":
                ser.write(b'\xC5')
            else:
                print("Use default mode")

            time.sleep(0.2)  # Wait before reading
            ser.flushInput()

            try:
                before = time.time()
                x = b''
                while len(x) < block_size:
                    x += ser.read(block_size - len(x))
                after = time.time()
            except Exception as e:
                print('Read Failed:', e)
                return None

            return x

    def _read_persistent(self, block_size, rng, _retry=False):
        """Optimized path using persistent connection. Skips open/close and soft reset.
        Only sends mode command when switching RNG source."""
        try:
            # Only send mode command if RNG source changed
            if rng != self._current_rng:
                if rng == "rng1":
                    self._ser.write(b'\xC4')
                elif rng == "rng2":
                    self._ser.write(b'\xC5')
                else:
                    print("Use default mode")
                time.sleep(0.2)
                self._ser.flushInput()
                self._current_rng = rng

            x = b''
            while len(x) < block_size:
                x += self._ser.read(block_size - len(x))
            return x

        except (serial.SerialException, OSError) as e:
            if _retry:
                print(f'Persistent read failed after reconnect: {e}')
                return None
            print(f'Serial error, attempting reconnect: {e}')
            self.close()
            try:
                self.open()
            except Exception:
                return None
            return self._read_persistent(block_size, rng, _retry=True)

if __name__ == '__main__':
    c = Connector()
    print(f" --- ")
    c.set_mode("MODE_UNWHITENED")
    uw = c.extract_lsb_stream_from_ascii(c.read(1024))
    uw1 = c.extract_lsb_stream_from_ascii(c.read(1024, "rng1"))
    print(f" --- ")
    uw2 = c.extract_lsb_stream_from_ascii(c.read(1024, "rng2"))
    print(f" --- ")
    c.set_mode("MODE_RAW_BIN")
    normal = c.extract_lsb_stream_from_binary(c.read(1024))
    print(" --- ")
    print(f" --- ")
    print(f"UW Z: {c.lsb_z_score(uw)}")
    print(f"UW1 Z: {c.lsb_z_score(uw1)}")
    print(f"UW2 Z: {c.lsb_z_score(uw2)}")
    print(f"Normal Z: {c.lsb_z_score(normal)}")
