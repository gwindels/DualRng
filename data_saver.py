import os
import csv
import pandas as pd
from datetime import datetime
from pytz import timezone


class DataSaver:
    def __init__(self, start_time, data_dir):
        self.start_time = start_time
        self.data_dir = data_dir
        self.cst = timezone('US/Central')

    def save_significant_events(self, significant_events):
        csv_filename = os.path.join(self.data_dir, f"significant_events_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv")
        df_events = pd.DataFrame(significant_events)
        file_exists = os.path.isfile(csv_filename)
        df_events.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
        # print(f"Significant events saved to {csv_filename}")

    def save_session_data(self, session_data):
        if not session_data:
            return
        start_time_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        stop_time_str = datetime.now(self.cst).strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"session_{start_time_str}_to_{stop_time_str}.csv")
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'timestamps', 
                'z_scores_1',
                'z_scores_2',
                'stouffers_z',
                'nvar',
                'svar'
            ])
            writer.writeheader()
            writer.writerows(session_data)
        # print(f"Session data saved to {filename}")

    def save_multiscale_events(self, events):
        """Save multi-scale significant events with window_scale column."""
        csv_filename = os.path.join(self.data_dir, f"multiscale_events_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv")
        df_events = pd.DataFrame(events)
        file_exists = os.path.isfile(csv_filename)
        df_events.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

    def save_correlation_values(self, df):
        cor_filename = os.path.join(self.data_dir, f"correlation_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv")
        cor_exists = os.path.isfile(cor_filename)
        df.to_csv(cor_filename, mode='a', header=not cor_exists, index=False)
        # print(f"Correlation values saved to {cor_filename}")

    def reset(self, new_start_time):
        self.start_time = new_start_time