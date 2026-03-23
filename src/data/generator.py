"""
Logic for generating realistic, causal time-series metrics with labeled incidents.
Models the "Four Golden Signals" for the first 3 metrics and fills the rest
with dynamic "Distractor" metrics to test the model's feature selection.
"""

import numpy as np
import pandas as pd

from src import config


def generate_synthetic_data(
    num_timesteps=config.TOTAL_TIMESTEPS,
    num_features=config.NUM_FEATURES,
    anomaly_prob=config.ANOMALY_PROB,
    anomaly_duration=config.ANOMALY_DURATION,
    random_state=config.RANDOM_STATE
):
    """
    Generates a synthetic dataset containing time-series metrics and incident labels.
    """
    np.random.seed(random_state)
    time_index = pd.date_range(start="2026-01-01", periods=num_timesteps, freq="min")
    df = pd.DataFrame({'timestamp': time_index})

    t = np.arange(num_timesteps)
    daily_cycle = np.sin(2 * np.pi * t / 1440)
    hourly_cycle = 0.5 * np.cos(2 * np.pi * t / 720)

    base_0 = None
    base_1 = None

    # metric_0: Traffic Metric
    if num_features > 0:
        noise_0 = np.random.normal(0, 50, num_timesteps)
        base_0 = 1000 + (400 * daily_cycle) + (100 * hourly_cycle) + noise_0
        df['metric_0'] = np.clip(base_0, 100, 5000)

    # metric_1: CPU Metric (Depends heavily on metric_0)
    if num_features > 1:
        noise_1 = np.random.normal(0, 2, num_timesteps)
        base_1 = 20 + (df['metric_0'] / 50) + noise_1
        df['metric_1'] = np.clip(base_1, 0, 100)

    # metric_2: Error Rate Metric (Mostly flat in a healthy system)
    if num_features > 2:
        base_2 = np.random.exponential(0.1, num_timesteps)
        df['metric_2'] = np.clip(base_2, 0, 100)

    # Generate Dynamic "Distractor" Metrics
    # These represent healthy background microservices. The ML model
    # must learn to ignore these to avoid false positives.
    for i in range(3, num_features):
        phase_shift = i * np.pi / 4
        distractor_cycle = np.sin(2 * np.pi * t / 1440 + phase_shift)
        distractor_noise = np.random.normal(0, 1.5, num_timesteps)
        df[f'metric_{i}'] = np.clip(40 + (15 * distractor_cycle) + distractor_noise, 0, 100)

    # The Incident Loop
    df['is_incident'] = 0
    potential_incident_starts = np.random.rand(num_timesteps) < anomaly_prob
    incident_indices = np.where(potential_incident_starts)[0]

    for start_idx in incident_indices:
        end_idx = min(start_idx + anomaly_duration, num_timesteps)

        # Label the entire incident duration as 1 (broken)
        df.loc[start_idx:end_idx - 1, 'is_incident'] = 1

        # Randomly select 1 of 3 realistic ways a server can break
        anomaly_type = np.random.choice(['ddos_spike', 'memory_leak', 'hardware_crash'])

        if anomaly_type == 'ddos_spike':
            # ----------------------------------------------------------------
            # PROFILE A: The DDoS Spike
            #   Traffic shoots up. The server struggles to keep up,
            #   eventually pinning the CPU to 100% and throwing errors.
            # ----------------------------------------------------------------
            lead_time = 20
            spike_start = max(0, start_idx - lead_time)

            # Traffic spikes 4 minutes before the crash
            if num_features > 0:
                df.loc[spike_start:end_idx - 1, 'metric_0'] *= np.random.uniform(3.0, 5.0)

            # CPU queues fill up and hit 100%
            if num_features > 1:
                df.loc[start_idx:end_idx - 1, 'metric_1'] = np.random.uniform(98, 100, end_idx - start_idx)

                # CPU gradually cools down after traffic dies down, but takes 10 minutes to fully recover
                recovery_end = min(num_timesteps, end_idx + 10)
                if end_idx < recovery_end:
                    # Linearly interpolate from 98% back down to the normal baseline over the recovery period
                    df.loc[end_idx:recovery_end - 1, 'metric_1'] = np.linspace(98, base_1[end_idx],
                                                                               recovery_end - end_idx)

            # Errors start showing up slightly after the CPU hits 100%
            if num_features > 2:
                error_start = min(start_idx + 1, end_idx - 1)
                if error_start < end_idx:
                    df.loc[error_start:end_idx - 1, 'metric_2'] += np.random.uniform(15, 35, end_idx - error_start)

        elif anomaly_type == 'memory_leak':
            # ----------------------------------------------------------------
            # PROFILE B: The Memory Leak
            #   CPU slowly drifts up over 25 minutes despite traffic
            #   being completely normal. Eventually, it crashes.
            # ----------------------------------------------------------------
            leak_duration = 25
            leak_start = max(0, start_idx - leak_duration)

            if num_features > 1:
                # CPU slowly drifts upward uncontrollably
                if leak_start < start_idx:
                    ramp = np.linspace(0, 60, start_idx - leak_start)
                    df.loc[leak_start:start_idx - 1, 'metric_1'] += ramp

                # System locks up, CPU hits 100%
                df.loc[start_idx:end_idx - 1, 'metric_1'] = np.random.uniform(95, 100, end_idx - start_idx)

                # Server rebooted, CPU drops to near zero briefly
                recovery_end = min(num_timesteps, end_idx + 5)
                if end_idx < recovery_end:
                    df.loc[end_idx:recovery_end - 1, 'metric_1'] = 10

            # Errors spike because the server is locked up
            if num_features > 2:
                df.loc[start_idx:end_idx - 1, 'metric_2'] += np.random.uniform(5, 20, end_idx - start_idx)

            # Traffic actually drops because the server stops responding to users
            if num_features > 0:
                df.loc[start_idx:end_idx - 1, 'metric_0'] *= 0.5

        elif anomaly_type == 'hardware_crash':
            # ----------------------------------------------------------------
            # PROFILE C: Hard Crash
            #   Someone trips over a power cord. The server vanishes instantly.
            #   There are 0 leading indicators. The ML model is meant to miss this.
            # ----------------------------------------------------------------

            # Traffic to this node goes to 0
            if num_features > 0:
                df.loc[start_idx:end_idx - 1, 'metric_0'] = 0

                # Traffic gradually recovers over 15 minutes
                recovery_end = min(num_timesteps, end_idx + 15)
                if end_idx < recovery_end:
                    df.loc[end_idx:recovery_end - 1, 'metric_0'] = np.linspace(0, base_0[end_idx],
                                                                               recovery_end - end_idx)

            # CPU goes to 0 because the box is off
            if num_features > 1:
                df.loc[start_idx:end_idx - 1, 'metric_1'] = 0

            # Load balancers throw massive errors trying to reach the dead node
            if num_features > 2:
                df.loc[start_idx:end_idx - 1, 'metric_2'] += np.random.uniform(80, 100, end_idx - start_idx)

    # Apply clipping to all metric columns to keep values physically realistic
    for col in df.columns:
        if col.startswith('metric_'):
            # metric_0 (Traffic) has a different scale (raw numbers) than percentages
            if col == 'metric_0':
                df[col] = np.clip(df[col], 0, None)
            else:
                df[col] = np.clip(df[col], 0, 100)

    return df


if __name__ == "__main__":
    print("Generating causal synthetic dataset...")
    df_synthetic = generate_synthetic_data()
    print(df_synthetic.head())

    incident_count = df_synthetic['is_incident'].sum()
    print(f"\nTotal time steps: {len(df_synthetic)}")
    print(
        f"Total time steps marked as incident: {incident_count} ({incident_count / len(df_synthetic) * 100:.2f}%)")
    print(f"Number of metrics generated: {config.NUM_FEATURES}")
