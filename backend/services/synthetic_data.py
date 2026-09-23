"""Generate teaching data, not a calibrated physical process simulation.

Each row is an independent operating snapshot, not a time-series sample.
The response includes noisy observations only, never the hidden clean target.
"""

import numpy as np

from backend.schemas import DatasetResponse, GenerationRequest, SensorRow


def generate_dataset(settings: GenerationRequest) -> DatasetResponse:
    """Combine five sensors using a documented nonlinear recipe and target noise.

    A local random generator avoids modifying global random state. Identical
    settings reproduce the same dataset in the same software environment.
    Noise is measured in percentage points of concentration, not relative %.
    """
    rng = np.random.default_rng(settings.seed)
    n = settings.n_samples
    temperature = rng.uniform(20, 90, n)
    density = rng.uniform(1000, 1300, n)
    flow = rng.uniform(10, 100, n)
    # A mild flow-pressure association makes sensor correlation visible.
    pressure = np.clip(1 + 0.04 * (flow - 10) + rng.normal(0, 0.4, n), 1, 6)
    agitation = rng.uniform(100, 800, n)

    # Center and scale into roughly [-1, 1], so coefficients are easy to read.
    d = (density - 1150) / 150
    t = (temperature - 55) / 35
    f = (flow - 55) / 45
    p = (pressure - 3.5) / 2.5
    a = (agitation - 450) / 350
    clean_target = (
        35 + 14 * d + 5 * t - 3 * f + 2 * p + 1.5 * a
        + 4 * d * t + 3 * t**2 - 2 * a**2
    )
    # Sample noise even at std=0 so the random draws have a consistent order.
    noisy_target = clean_target + settings.noise_std * rng.normal(0, 1, n)
    clipped_count = int(np.count_nonzero((noisy_target < 0) | (noisy_target > 100)))
    target = np.clip(noisy_target, 0, 100)

    values = zip(temperature, density, flow, pressure, agitation, target)
    rows = [SensorRow(
        temperature_c=round(float(temp), 4),
        density_kg_m3=round(float(dens), 4),
        flow_rate_l_min=round(float(rate), 4),
        pressure_bar=round(float(pres), 4),
        agitation_rpm=round(float(speed), 4),
        solid_concentration_pct=round(float(concentration), 4),
    ) for temp, dens, rate, pres, speed, concentration in values]
    return DatasetResponse(settings=settings, clipped_target_count=clipped_count, rows=rows)
