from math import ceil
from typing import Tuple

import numpy as np
import rasterio

from eis_toolkit.checks.crs import check_matching_crs
from eis_toolkit.exceptions import (
    InvalidPixelSizeException,
    NonMatchingCrsException,
    NonSquarePixelSizeException,
)


# The core snapping functionality. Used internally by snap.
def _snap(  # type: ignore[no-any-unimported]
    raster: rasterio.DatasetReader, snap_raster: rasterio.DatasetReader
) -> Tuple[np.ndarray, dict]:
    raster_bounds = raster.bounds
    snap_bounds = snap_raster.bounds
    raster_pixel_size = raster.transform.a
    snap_pixel_size = snap_raster.transform.a

    cells_added = ceil(snap_pixel_size / raster_pixel_size)

    out_image = np.full((raster.count, raster.height + cells_added, raster.width + cells_added), raster.nodata)
    out_meta = raster.meta.copy()

    # Coordinates for the snap raster boundaries
    left_distance_in_pixels = (raster_bounds.left - snap_bounds.left) // snap_pixel_size
    left_snap_coordinate = snap_bounds.left + left_distance_in_pixels * snap_pixel_size

    bottom_distance_in_pixels = (raster_bounds.bottom - snap_bounds.bottom) // snap_pixel_size
    bottom_snap_coordinate = snap_bounds.bottom + bottom_distance_in_pixels * snap_pixel_size
    top_snap_coordinate = bottom_snap_coordinate + (raster.height + cells_added) * raster_pixel_size

    # Distance and array indices of close cell corner in snapped raster to slot values
    x_distance = (raster_bounds.left - left_snap_coordinate) % raster_pixel_size
    x0 = int((raster_bounds.left - left_snap_coordinate) // raster_pixel_size)
    x1 = x0 + raster.width

    y_distance = (raster_bounds.bottom - bottom_snap_coordinate) % raster_pixel_size
    y0 = int(cells_added - ((raster_bounds.bottom - bottom_snap_coordinate) // raster_pixel_size))
    y1 = y0 + raster.height

    # Find the closest corner of the snapped grid for shifting/slotting the original raster
    if x_distance < raster_pixel_size / 2 and y_distance < raster_pixel_size / 2:
        out_image[:, y0:y1, x0:x1] = raster.read()  # Snap values towards left-bottom
    elif x_distance < raster_pixel_size / 2 and y_distance > raster_pixel_size / 2:
        out_image[:, y0 - 1 : y1 - 1, x0:x1] = raster.read()  # Snap values towards left-top # noqa: E203
    elif x_distance > raster_pixel_size / 2 and y_distance > raster_pixel_size / 2:
        out_image[:, y0 - 1 : y1 - 1, x0 + 1 : x1 + 1] = raster.read()  # Snap values toward right-top # noqa: E203
    else:
        out_image[:, y0:y1, x0 + 1 : x1 + 1] = raster.read()  # Snap values towards right-bottom # noqa: E203

    out_transform = rasterio.Affine(
        raster.transform.a,
        raster.transform.b,
        left_snap_coordinate,
        raster.transform.d,
        raster.transform.e,
        top_snap_coordinate,
    )
    out_meta.update({"transform": out_transform, "width": out_image.shape[-1], "height": out_image.shape[-2]})
    return out_image, out_meta


def snap_with_raster(  # type: ignore[no-any-unimported]
    raster: rasterio.DatasetReader, snap_raster: rasterio.DatasetReader
) -> Tuple[np.ndarray, dict]:
    """Snaps/aligns raster to given snap raster.

    Raster is snapped from its left-bottom corner to nearest snap raster grid corner in left-bottom direction.
    Both raster and snap raster need to have square-shaped pixels.
    If rasters are aligned, simply returns input raster data and metadata.

    Args:
        raster (rasterio.io.DatasetReader): The raster to be clipped.
        snap_raster (rasterio.io.DatasetReader): The snap raster i.e. reference grid raster.

    Returns:
        out_image (np.ndarray): The snapped raster data.
        out_meta (dict): The updated metadata.

    Raises:
        NonMatchingCrsException: Raster and and snap raster are not in the same crs.
        NonSquarePixelSizeException: Raster or snap raster has nonsquare pixels.
    """

    if not check_matching_crs(
        objects=[raster, snap_raster],
    ):
        raise NonMatchingCrsException

    # Account for small rounding errors if raster has been resampled
    if (
        not abs(raster.transform.a + raster.transform.e) < 0.00001
        or not abs(snap_raster.transform.a + snap_raster.transform.e) < 0.00001
    ):
        raise NonSquarePixelSizeException

    if snap_raster.bounds.bottom == raster.bounds.bottom and snap_raster.bounds.left == raster.bounds.left:
        out_image, out_meta = raster.read(), raster.meta
        return out_image, out_meta

    out_image, out_meta = _snap(raster, snap_raster)
    return out_image, out_meta