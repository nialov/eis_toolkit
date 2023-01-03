from typing import List, Optional

import numpy as np
import os
import pandas as pd
import rasterio

def idw(
    row: np.float64,
    column: np.float64,
    values: List[pd.core.series.Series],
    rows: List[pd.core.series.Series],
    columns: List[pd.core.series.Series],
    power: int
) -> np.float64:

    results = np.empty(shape=( len(rows), 6))
    results[:, 0] = rows
    results[:, 1] = columns
    results[:, 2] = values

    # weight = 1 / (d(x, x_i)^power + 1)
    results[:, 3] = 1 / (np.sqrt((results[:, 0] - row)**2 + (results[:, 1] - column)**2)**power + 1)

    # Multiplicative product of inverse distant weight and actual value.
    results[:, 4] = results[:, 2] * results[:, 3]

    # Divide sum of weighted value by sum of weights to get IDW interpolation.
    idw = results[:, 4].sum() / results[:, 3].sum()
    return idw


def _data_frame_to_idw_raster(
    base_raster: rasterio.io.DatasetReader,
    data_frame: pd.DataFrame,
    output_file: str,
    power: Optional[int] = 2
) -> rasterio.io.DatasetReader:

    height = base_raster.meta['height']
    width = base_raster.meta['width']

    raster = np.empty((height, width))
    raster[:] = np.nan
    raster_transform = base_raster.meta['transform']

    value_columns = data_frame.loc[:, ~data_frame.columns.isin(['x', 'y'])]

    for value_column in value_columns:
        values = value_columns[value_column]

        for row in range(height):
            for column in range(width):

                x, y = rasterio.transform.xy(raster_transform, row, column)

                value = idw(
                            row = x,
                            column = y,
                            values = values,
                            rows = data_frame['x'],
                            columns = data_frame['y'],
                            power = power
                            )

                raster[row, column] = value

    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'transform': raster_transform,
        'crs': base_raster.meta['crs'],
        'count': 1,
        'dtype': base_raster.meta['dtype']
    }

    with rasterio.open(output_file, 'w', **meta) as output_raster:
        output_raster.write(raster, 1)

    return rasterio.open(output_file)


def data_frame_to_idw_raster(
    base_raster: rasterio.io.DatasetReader,
    data_frame: pd.DataFrame,
    output_file: str,
    power: Optional[int] = 2
) -> rasterio.io.DatasetReader:
    """Creates an Inverse Distance Weighted (IDW) interpolation using a base raster and point data.
        
        The algorithm interpolates using point data values to all cells of the base raster.
        Power parameter defines the emphasis of nearby points.
        As power increases, interpolated values begin to approach value of the nearest point.
        Weight is calculated using the following formula: w = 1 / (d(x, y)^power + 1).
        The formula is derived from pyidw library by Tamim & Popy (2022) http://dx.doi.org/10.2139/ssrn.4123955.
        
    Args:
        base_raster (rasterio.io.DatasetReader]: Bounds of the interpolation area.
        data_frame (pd.DataFrame): Points to interpolate values with.
        output_file (str): Output file destination.
        power (int, optional): Defines the emphasis of nearest points. Defaults to 2.
    Returns:
        rasterio.io.DatasetReader: 
    """

    if not isinstance(base_raster, rasterio.io.DatasetReader):
        raise InvalidParameterValueException
    if not isinstance(data_frame, pd.DataFrame):
        raise InvalidParameterValueException
    if not isinstance(output_file, str):
        raise InvalidParameterValueException
    directory_of_file = os.path.dirname(os.path.abspath(output_file))
    if not os.path.exists(directory_of_file):
        raise InvalidParameterValueException    
    if not isinstance(power, int):
        raise InvalidParameterValueException

    output_raster = _data_frame_to_idw_raster(
        base_raster = base_raster,
        data_frame = data_frame,
        output_file = output_file,
        power = power)

    return output_raster