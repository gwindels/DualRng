import numpy as np

class NetVarCalculator:
    def __init__(self, min_raw=0, max_raw=6):
        """Initialize NetVar calculator with fixed scaling range."""
        self.min_raw = min_raw  # Fixed lower bound for raw NetVar
        self.max_raw = max_raw  # Fixed upper bound for raw NetVar

    def compute_stouffers_z(self, *z_score_lists):
        """Computes Stouffer’s Z-score from multiple independent Z-score series."""
        z_array = np.array(z_score_lists)
        return np.sum(z_array, axis=0) / np.sqrt(z_array.shape[0])  # Normalizes correctly

    def compute_netvar(self, stouffers_z):
        """Computes NetVar as a Chi-square statistic with 1 d.f."""
        return stouffers_z ** 2  

    def scale_to_netvar(self, netvar):
        """Scales NetVar into the 1–324 range using a fixed range."""
        min_scaled, max_scaled = 1, 324  
        return int(round(np.interp(netvar, (self.min_raw, self.max_raw), (min_scaled, max_scaled))))

    def categorize_netvar(self, netvar):
        """Categorizes NetVar based on GCP2's scale."""
        if 1 <= netvar < 140:
            return "Normal"
        elif 140 <= netvar < 166:
            return "Elevated"
        elif 166 <= netvar < 219:
            return "High"
        elif 219 <= netvar < 279:
            return "Very High"
        elif 279 <= netvar <= 324:
            return "Extreme"
        return "Unknown"
    
    def reset(self, min_raw=0, max_raw=6):
        self.min_raw = min_raw  # Fixed lower bound for raw NetVar
        self.max_raw = max_raw  # Fixed upper bound for raw NetVar