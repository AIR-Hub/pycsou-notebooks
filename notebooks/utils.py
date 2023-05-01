import glob
import io
import os
import urllib.request as urlrequest
import zipfile
from typing import Tuple

import imageio.v2 as iio
import numpy as np
from scipy.ndimage import zoom


def get_epfl_deconv_data():
    """Download example data from EPFL Biomedical Imaging Group.

    Download deconvolution problem data from EPFL Biomedical Imaging
    Group. The downloaded data is converted to `.npz` format for
    convenient access via :func:`numpy.load`. The converted data is saved
    in a file `epfl_big_deconv_<channel>.npz` in the directory specified
    by `path`.

    Args:
        channel: Channel number between 0 and 2.
        path: Directory in which converted data is saved.
        verbose: Flag indicating whether to print status messages.
    """

    # data source URL and filenames
    data_base_url = "http://bigwww.epfl.ch/deconvolution/bio/"
    data_zip_files = ["CElegans-CY3.zip", "CElegans-DAPI.zip", "CElegans-FITC.zip"]
    psf_zip_files = ["PSF-" + data for data in data_zip_files]

    # ensure path directory exists
    path = "data/epfl_data/"
    if not os.path.exists(path):
        os.makedirs(path)
    for channel in range(3):
        # download data and psf files for selected channel into temporary directory
        for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
            print(f"Downloading {zip_file} from {data_base_url}")
            data = url_get(data_base_url + zip_file)
            f = open(os.path.join(path, zip_file), "wb")
            f.write(data.read())
            f.close()

        # unzip downloaded data into temporary directory
        for zip_file in (data_zip_files[channel], psf_zip_files[channel]):
            print(f"Extracting content from zip file {zip_file}")
            with zipfile.ZipFile(os.path.join(path, zip_file), "r") as zip_ref:
                zip_ref.extractall(path)

        # read unzipped data files into 3D arrays and save as .npz
        zip_file = data_zip_files[channel]
        y = volume_read(os.path.join(path, zip_file[:-4]))
        zip_file = psf_zip_files[channel]
        psf = volume_read(os.path.join(path, zip_file[:-4]))

        npz_file = os.path.join(path, f"epfl_big_deconv_{channel}.npz")
        print(f"Saving as {npz_file}")

        np.savez(npz_file, y=y, psf=psf)
    print("Download complete")


def epfl_deconv_data(channel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get deconvolution problem data from EPFL Biomedical Imaging Group.

    If the data has previously been downloaded, it will be retrieved from
    a local data/epfl_data folder.

    Args:
        channel: Channel number between 0 and 2.

    Returns:
       tuple: A tuple (y, psf) containing:

           - **y** : (np.array): Blurred channel data.
           - **psf** : (np.array): Channel psf.
    """

    path = "data/epfl_data/"

    # create cache directory and download data if not already present
    try:
        npz_file = os.path.join(path, f"epfl_big_deconv_{channel}.npz")
        npz = np.load(npz_file)
    except:
        get_epfl_deconv_data()
        npz_file = os.path.join(path, f"epfl_big_deconv_{channel}.npz")
        npz = np.load(npz_file)
    # load data and return y and psf arrays converted to float32
    y = npz["y"].astype(np.float32)
    psf = npz["psf"].astype(np.float32)
    return y, psf


def url_get(
    url: str, maxtry: int = 3, timeout: int = 10
) -> io.BytesIO:  # pragma: no cover
    """Get content of a file via a URL.

    Args:
        url: URL of the file to be downloaded.
        maxtry: Maximum number of download retries.
        timeout: Timeout in seconds for blocking operations.

    Returns:
        Buffered I/O stream.

    Raises:
        ValueError: If the maxtry parameter is not greater than zero.
        urllib.error.URLError: If the file cannot be downloaded.
    """

    if maxtry <= 0:
        raise ValueError("Parameter maxtry should be greater than zero.")
    for ntry in range(maxtry):
        try:
            rspns = urlrequest.urlopen(url, timeout=timeout)
            cntnt = rspns.read()
            break
        except urlerror.URLError as e:
            if not isinstance(e.reason, socket.timeout):
                raise

    return io.BytesIO(cntnt)


def volume_read(path: str, ext: str = "tif") -> np.ndarray:
    """Read a 3D volume from a set of files in the specified directory.

    All files with extension `ext` (i.e. matching glob `*.ext`)
    in directory `path` are assumed to be image files and are read.
    The filenames are assumed to be such that their alphanumeric
    ordering corresponds to their order as volume slices.

    Args:
        path: Path to directory containing the image files.
        ext: Filename extension.

    Returns:
        Volume as a 3D array.
    """

    slices = []
    for file in sorted(glob.glob(os.path.join(path, "*." + ext))):
        image = iio.imread(file)
        slices.append(image)
    return np.dstack(slices)


def downsample_volume(vol, rate):
    """Downsample a 3D array.

    Downsample a 3D array. If the volume dimensions can be divided by
    `rate`, this is achieved via averaging distinct `rate` x `rate` x
    `rate` block in `vol`. Otherwise it is achieved via a call to
    :func:`scipy.ndimage.zoom`.

    Args:
        vol: Input volume.
        rate: Downsampling rate.

    Returns:
        Downsampled volume.
    """

    if rate == 1:
        return vol

    if np.all([n % rate == 0 for n in vol.shape]):
        vol = np.mean(np.reshape(vol, (-1, rate, vol.shape[1], vol.shape[2])), axis=1)
        vol = np.mean(np.reshape(vol, (vol.shape[0], -1, rate, vol.shape[2])), axis=2)
        vol = np.mean(np.reshape(vol, (vol.shape[0], vol.shape[1], -1, rate)), axis=3)
    else:
        vol = zoom(vol, 1.0 / rate)

    return vol
