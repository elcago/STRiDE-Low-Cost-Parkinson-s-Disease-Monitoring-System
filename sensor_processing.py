import argparse
import pandas as pd
import numpy as np

FIXED_RESISTOR_OHM = 100_000.0
ADC_MAX = 1023.0
CALIBRATION_SLOPE = 0.00071
CALIBRATION_INTERCEPT = 0.000801


def adc_to_resistance(sensor_value):
    sensor_value = np.asarray(sensor_value, dtype=float)
    sensor_value = np.clip(sensor_value, 1, ADC_MAX - 1)
    return FIXED_RESISTOR_OHM / ((ADC_MAX / sensor_value) - 1)


def adc_to_weight_kg(sensor_value):
    resistance = adc_to_resistance(sensor_value)
    conductance = 1.0 / resistance
    weight = (conductance - CALIBRATION_INTERCEPT) / CALIBRATION_SLOPE
    return np.maximum(weight, 0.0)


def convert_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    sensor_cols = [c for c in df.columns if c.lower().startswith("sensor")]

    if not sensor_cols:
        raise ValueError("No sensor columns found. Expected names like sensor1, sensor2, ...")

    for col in sensor_cols:
        df[f"{col}_kg"] = adc_to_weight_kg(df[col].to_numpy())

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    convert_file(args.input_csv, args.output_csv)
