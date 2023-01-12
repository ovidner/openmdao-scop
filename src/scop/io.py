from numbers import Number

import jsonpickle
import numpy as np
import xarray as xr

CURRENT_ENCODING_VERSION = 0


def encode_attrs(attrs):
    valid_types = (str, Number, np.ndarray, np.number, list, tuple, np.bool_)
    encoded_keys = [
        key for key, value in attrs.items() if not isinstance(value, valid_types)
    ]

    enc_attrs = {
        key: jsonpickle.encode(value) if key in encoded_keys else value
        for key, value in attrs.items()
    }

    return {
        **enc_attrs,
        "_scop:encoded_keys": encoded_keys,
    }


def decode_attrs(attrs):
    encoded_keys = attrs.pop("_scop:encoded_keys")
    dec_attrs = {
        key: jsonpickle.decode(value) if key in encoded_keys else value
        for key, value in attrs.items()
    }

    return dec_attrs


def jsonencode_attrs(attrs):
    return {"_scop:encoded_attrs": jsonpickle.encode(attrs)}


def jsondecode_attrs(attrs):
    return jsonpickle.decode(attrs["_scop:encoded_attrs"])


def dump_netcdf(ds: xr.Dataset, path, default_compression="lzf", **kwargs):
    # Make a shallow copy so we don't mangle the attrs of the ds we're dumping.
    enc_ds = ds.copy(deep=False)

    enc_ds.attrs = jsonencode_attrs(enc_ds.attrs)

    for var in enc_ds.variables.values():
        var.attrs = jsonencode_attrs(var.attrs)
        if default_compression:
            var.encoding.setdefault("compression", default_compression)

    enc_ds.attrs["_scop:encoding_version"] = CURRENT_ENCODING_VERSION

    return enc_ds.to_netcdf(path=path, engine="h5netcdf", invalid_netcdf=True, **kwargs)


def dump_zarr(ds, path, **kwargs):
    # Make a shallow copy so we don't mangle the attrs of the ds we're dumping.
    enc_ds = ds.copy(deep=False)

    _unsafe_var_names = [(name, name.replace(":", ".")) for name in enc_ds.variables if ":" in name]
    unsafe_var_names_fw = {orig: new for orig, new in _unsafe_var_names}
    unsafe_var_names_bw = {new: orig for orig, new in _unsafe_var_names}

    # Make sure we don't have any name collisions.
    assert len(unsafe_var_names_fw) == len(unsafe_var_names_bw)

    enc_ds = enc_ds.rename(unsafe_var_names_fw)
    enc_ds.attrs["_scop:unsafe_var_names"] = unsafe_var_names_bw

    enc_ds.attrs = jsonencode_attrs(enc_ds.attrs)

    for var in enc_ds.variables.values():
        var.attrs = jsonencode_attrs(var.attrs)

    enc_ds.attrs["_scop:encoding_version"] = CURRENT_ENCODING_VERSION

    return enc_ds.to_zarr(path, **kwargs)


def _load_postprocess(ds):
    encoding_version = ds.attrs.pop("_scop:encoding_version", None)
    if encoding_version is None:
        raise ValueError("File cannot be recognized as a Scop dataset.")
    elif encoding_version > CURRENT_ENCODING_VERSION:
        raise ValueError(
            f"Dataset has a too new version ({encoding_version}). Maximum supported version is {CURRENT_ENCODING_VERSION}."
        )

    ds.attrs = jsondecode_attrs(ds.attrs)
    for var in ds.variables.values():
        var.attrs = jsondecode_attrs(var.attrs)

    unsafe_var_names = ds.attrs.pop("_scop:unsafe_var_names", None)
    if unsafe_var_names:
        ds = ds.rename(unsafe_var_names)

    return ds


def load_netcdf(path, **kwargs):
    ds = xr.open_dataset(path, engine="h5netcdf", **kwargs)

    return _load_postprocess(ds)


def load_zarr(path, **kwargs):
    ds = xr.open_zarr(path, **kwargs)

    return _load_postprocess(ds)


dump = dump_zarr
load = load_zarr
