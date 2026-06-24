from copy import deepcopy
from typing import List

from .filedata.abstract import FileData, FileMetaData
from .filedata.binary import BinaryFile
from .filedata.ildg import IldgFile
from .filedata.timeslice import QDPLazyDiskMapObjFile
from .filedata.ndarray import NdarrayFile, NdarrayTimeslicesFile


class GaugeField:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class Eigenvector:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class Elemental:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class CurrentElemental:
    """
    Base class for current elemental data loaders.

    Similar to Elemental but for current vertex data that may include point information.
    """

    def __init__(self, elem: FileMetaData, eigenNum: int, pointNum: int = None) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum
        self.Np = pointNum


class Perambulator:
    def __init__(self, elem: FileMetaData, eigenNum: int) -> None:
        self.elem = deepcopy(elem)
        self.Ne = eigenNum


class PointSource:
    def __init__(self, elem: FileMetaData, Np: int) -> None:
        self.elem = deepcopy(elem)
        self.Np = Np


class OverlapMatrix:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class PropagatorPSV:
    """
    Point-to-eigenvector propagator (PSV).

    Full shape: [Lt, Lt, Ns, Ns, Np, Nc, Ne] for single file version
                [Lt, Ns, Ns, Np, Nc, Ne] for timeslice version

    PSV[t_snk, t_src, s_snk, s_src, p, c, e] represents:
        <eigenvector_e(c) | S(x_p, t_snk; source, t_src) | source>
    where:
        - S is the Dirac propagator
        - x_p is the point source position
        - c is the color index
        - e is the eigenvector index
    """

    def __init__(self, elem: FileMetaData, Np: int, Ne: int) -> None:
        self.elem = deepcopy(elem)
        self.Np = Np
        self.Ne = Ne


class PropagatorVSP:
    """
    Eigenvector-to-point propagator (VSP).

    Full shape: [Lt, Lt, Ns, Ns, Ne, Np, Nc] for single file version
                [Lt, Ns, Ns, Ne, Np, Nc] for timeslice version

    VSP[t_snk, t_src, s_snk, s_src, e, p, c] represents:
        <point_p(c) | S(x_snk, t_snk; eigenvector_e, t_src) | eigenvector_e>
    where:
        - S is the Dirac propagator
        - x_snk is the point sink position
        - c is the color index
        - e is the eigenvector index

    Note: VSP and PSV have the same shape but represent different directions of propagation.
    """

    def __init__(self, elem: FileMetaData, Np: int, Ne: int) -> None:
        self.elem = deepcopy(elem)
        self.Np = Np
        self.Ne = Ne


class PropagatorPSP:
    """
    Point-to-point propagator (PSP).

    Full shape: [Lt, Lt, Ns, Ns, Np_snk, Nc_snk, Np_src, Nc_src] for single file version
                [Lt, Ns, Ns, Np_snk, Nc_snk, Np_src, Nc_src] for timeslice version

    PSP[t_snk, t_src, s_snk, s_src, p_snk, c_snk, p_src, c_src] represents:
        <point_p_snk(c) | S(x_snk, t_snk; x_src, t_src) | point_p_src>
    where:
        - S is the Dirac propagator
        - x_snk, x_src are point sink and source positions
        - c_snk, c_src are color indices
        - p_snk, p_src are point indices
    """

    def __init__(self, elem: FileMetaData, Np_snk: int, Np_src: int) -> None:
        self.elem = deepcopy(elem)
        self.Np_snk = Np_snk
        self.Np_src = Np_src


class OnePoint:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class TwoPoint:
    def __init__(self, elem: FileMetaData) -> None:
        self.elem = deepcopy(elem)


class GaugeFieldTimeSlice(QDPLazyDiskMapObjFile, GaugeField):
    def __init__(
        self, prefix: str, suffix: str, shape: List[int] = [128, 4, 16**3, 3, 3]
    ) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, ">c16", 2))
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.mod" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenvectorTimeSlice(QDPLazyDiskMapObjFile, Eigenvector):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [128, 70, 16**3, 3],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Eigenvector.__init__(self, FileMetaData(shape, ">c8", 2), totNe)
        self.prefix = prefix
        self.suffix = (
            ".stout.n20.f0.12.laplace_eigs.3d.mod" if suffix is None else suffix
        )

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class EigenvectorNpy(NdarrayFile, Eigenvector):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [70, 128, 16**3, 3],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Eigenvector.__init__(self, FileMetaData(shape, "<c16", 2), totNe)
        self.prefix = prefix
        self.suffix = ".lime.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorBinary(BinaryFile, Perambulator):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [128, 128, 4, 4, 70, 70],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorNpy(NdarrayFile, Perambulator):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [128, 128, 4, 4, 70, 70],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PointSourceNpy(NdarrayFile, PointSource):
    def __init__(
        self, prefix: str, suffix: str, shape: List[int] = [128, 72, 3], Np: int = 128
    ) -> None:
        super().__init__()
        PointSource.__init__(self, FileMetaData(shape, "<c8", 0), Np)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class OverlapMatrixNpy(NdarrayFile, OverlapMatrix):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [72, 128, 216, 3],  # [Lt, Ne, Np, Nc]
        Ne: int = 128,
        Np: int = 216,
    ) -> None:
        super().__init__()
        OverlapMatrix.__init__(self, FileMetaData(shape, "<c16", 2))
        self.prefix = prefix
        self.suffix = ".overlap_matrix.npy" if suffix is None else suffix
        self.Ne = Ne
        self.Np = Np

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorPSVNpy(NdarrayFile, PropagatorPSV):
    """
    Load PSV propagator from a single .npy file.

    This class loads the complete PSV propagator data with both source and sink time dimensions.

    Parameters:
    -----------
    prefix : str
        File path prefix
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Data shape: [Lt, Lt, Ns, Ns, Np, Nc, Ne]
        where:
            Lt: temporal extent (both source and sink times)
            Ns: Dirac spin dimension (4)
            Np: number of point sources
            Nc: number of colors (3)
            Ne: number of eigenvectors
    Np : int
        Number of point sources
    Ne : int
        Number of eigenvectors
    dtype : str
        Data type, default "<c16" (complex128)

    Example:
    --------
    >>> # Standard shape: [Lt, Lt, Ns, Ns, Np, Nc, Ne]
    >>> psv = PropagatorPSVNpy(
    ...     prefix="/path/to/data/cfg_",
    ...     suffix=".psv.npy",
    ...     shape=[72, 72, 4, 4, 216, 3, 70],
    ...     Np=216,
    ...     Ne=70
    ... )
    >>> data = psv.load("1000")  # loads cfg_1000.psv.npy
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np: int,
        Ne: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorPSV.__init__(self, FileMetaData(shape, dtype, 0), Np, Ne)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorVSPNpy(NdarrayFile, PropagatorVSP):
    """
    Load VSP propagator from a single .npy file.

    This class loads the complete VSP (eigenvector-to-point) propagator data.

    Parameters:
    -----------
    prefix : str
        File path prefix
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Data shape: [Lt, Lt, Ns, Ns, Ne, Np, Nc]
        where:
            Lt: temporal extent (both source and sink times)
            Ns: Dirac spin dimension (4)
            Ne: number of eigenvectors (sources)
            Np: number of point sinks
            Nc: number of colors (3)
    Np : int
        Number of point sinks
    Ne : int
        Number of eigenvectors
    dtype : str
        Data type, default "<c16" (complex128)

    Example:
    --------
    >>> # Standard shape: [Lt, Lt, Ns, Ns, Ne, Np, Nc]
    >>> vsp = PropagatorVSPNpy(
    ...     prefix="/path/to/data/cfg_",
    ...     suffix=".vsp.npy",
    ...     shape=[72, 72, 4, 4, 216, 3, 70],
    ...     Np=216,
    ...     Ne=70
    ... )
    >>> data = vsp.load("1000")
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np: int,
        Ne: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorVSP.__init__(self, FileMetaData(shape, dtype, 0), Np, Ne)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorVSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorVSP):
    """
    Load VSP propagator from timeslice-separated .npy files.

    File naming convention: {prefix}{cfg}.t{t_src:03d}{suffix}

    Parameters:
    -----------
    prefix : str
        File path prefix
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Shape of FULL propagator: [Lt, Lt, Ns, Ns, Ne, Np, Nc]
        Each file has shape: [Lt, Ns, Ns, Ne, Np, Nc]
    Np : int
        Number of point sinks
    Ne : int
        Number of eigenvectors
    dtype : str
        Data type (default: "<c16")

    Example:
    --------
    >>> vsp = PropagatorVSPTimeslicesNpy(
    ...     prefix="/path/to/data/cfg_",
    ...     suffix=".npy",
    ...     shape=[72, 72, 4, 4, 216, 3, 70],
    ...     Np=216,
    ...     Ne=70
    ... )
    >>> data = vsp.load("1000")
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np: int,
        Ne: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorVSP.__init__(self, FileMetaData(shape, dtype, 0), Np, Ne)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorPSPNpy(NdarrayFile, PropagatorPSP):
    """
    Load PSP propagator from a single .npy file.

    This class loads the complete PSP (point-to-point) propagator data.

    Parameters:
    -----------
    prefix : str
        File path prefix
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Data shape: [Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]
        where:
            Lt: temporal extent (both source and sink times)
            Ns: Dirac spin dimension (4)
            Np_snk: number of point sinks
            Nc: number of colors (3)
            Np_src: number of point sources
    Np_snk : int
        Number of point sinks
    Np_src : int
        Number of point sources
    dtype : str
        Data type, default "<c16" (complex128)

    Example:
    --------
    >>> # Standard shape: [Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
    >>> psp = PropagatorPSPNpy(
        ...     prefix="/path/to/data/cfg_",
        ...     suffix=".psp.npy",
        ...     shape=[72, 72, 4, 4, 216, 3, 216, 3],
        ...     Np_snk=216,
        ...     Np_src=216
        ... )
    >>> data = psp.load("1000")
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 216)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np_snk: int,
        Np_src: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorPSP.__init__(self, FileMetaData(shape, dtype, 0), Np_snk, Np_src)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorPSPTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSP):
    """
    Load PSP propagator from timeslice-separated .npy files.

    File naming convention: {prefix}{cfg}.t{t_src:03d}{suffix}

    Parameters:
    -----------
    prefix : str
        File path prefix
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Shape of FULL propagator: [Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
        Each file has shape: [Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc]
    Np_snk : int
        Number of point sinks
    Np_src : int
        Number of point sources
    dtype : str
        Data type (default: "<c16")

    Example:
    --------
    >>> psp = PropagatorPSPTimeslicesNpy(
        ...     prefix="/path/to/data/cfg_",
        ...     suffix=".npy",
        ...     shape=[72, 72, 4, 4, 216, 3, 216, 3],
        ...     Np_snk=216,
        ...     Np_src=216
        ... )
    >>> data = psp.load("1000")
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 216)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np_snk: int,
        Np_src: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorPSP.__init__(self, FileMetaData(shape, dtype, 0), Np_snk, Np_src)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PerambulatorTimeslicesNpy(NdarrayTimeslicesFile, Perambulator):
    """
    this Perambulator data class is modified for timeslide solely saved  data.
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [128, 128, 4, 4, 70, 70],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class PropagatorPSVTimeslicesNpy(NdarrayTimeslicesFile, PropagatorPSV):
    """
    Load PSV propagator from timeslice-separated .npy files.

    This class is designed for PSV data saved per source time slice, where each file
    contains data for a specific source time with one less time dimension than the
    complete propagator.

    File naming convention: {prefix}{cfg}.t{t_src:03d}{suffix}
    Example: /path/to/data/cfg_1000.t000.npy, cfg_1000.t001.npy, ...

    Parameters:
    -----------
    prefix : str
        File path prefix (including directory and cfg prefix)
    suffix : str
        File suffix (default: ".npy")
    shape : List[int]
        Shape of FULL propagator (after loading all timeslices): [Lt, Lt, Ns, Ns, Np, Nc, Ne]
        Each individual file has shape: [Lt, Ns, Ns, Np, Nc, Ne]
        where:
            Lt: temporal extent
            Ns: Dirac spin dimension (4)
            Np: number of point sources
            Nc: number of colors (3)
            Ne: number of eigenvectors
    Np : int
        Number of point sources
    Ne : int
        Number of eigenvectors
    dtype : str
        Data type (default: "<c16")

    Notes:
    ------
    Each timeslice file has shape [Lt, Ns, Ns, Np, Nc, Ne], representing:
        PSV[t_snk, s_snk, s_src, p, c, e] for fixed t_src

    When all files are loaded, they are assembled into shape [Lt, Lt, Ns, Ns, Np, Nc, Ne]
    where the first Lt is t_src (source time) and second Lt is t_snk (sink time).

    Example:
    --------
    >>> # Standard usage with timeslice-separated files
    >>> psv = PropagatorPSVTimeslicesNpy(
    ...     prefix="/path/to/data/cfg_",
    ...     suffix=".npy",
    ...     shape=[72, 72, 4, 4, 216, 3, 70],
    ...     Np=216,
    ...     Ne=70
    ... )
    >>> # Load configuration "1000"
    >>> # Loads: cfg_1000.t000.npy, cfg_1000.t001.npy, ..., cfg_1000.t071.npy
    >>> # Each file has shape [72, 4, 4, 216, 3, 70]
    >>> data = psv.load("1000")
    >>> print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)

    Generation example (from gen_propagator.py):
    >>> for t_src in range(Lt):
    ...     VSV, PSV, VSP, PSP = perambulator.calc(t_src)
    ...     peramb_PSV = np.roll(PSV.get(), -t_src, 0)
    ...     # PSV.get() has shape [Lt, Ns, Ns, Np, Nc, Ne]
    ...     np.save(f"{save_dir}/cfg_{cfg}.t{t_src:03d}.npy", peramb_PSV)
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],
        Np: int,
        Ne: int,
        dtype: str = "<c16",
    ) -> None:
        super().__init__()
        PropagatorPSV.__init__(self, FileMetaData(shape, dtype, 0), Np, Ne)
        self.prefix = prefix
        self.suffix = ".npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalBinary(BinaryFile, Elemental):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [40, 27, 128, 70, 70],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Elemental.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaBinary(BinaryFile, TwoPoint):
    def __init__(
        self, prefix: str, suffix: str, shape: List[int] = [128, 2, 3, 4, 27, 128]
    ) -> None:
        super().__init__()
        TwoPoint.__init__(self, FileMetaData(shape, "<f8", 0))
        self.prefix = prefix
        self.suffix = ".mesonspec.2pt.bin" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class GaugeFieldIldg(IldgFile, GaugeField):
    def __init__(
        self, prefix: str, suffix: str, shape: List[int] = [128, 16**3, 4, 3, 3]
    ) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, ">c16", 0))
        self.prefix = prefix
        self.suffix = ".lime" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class GaugeFieldBinary(BinaryFile, GaugeField):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [128, 16**3, 4, 3, 3],
        dtype: str = "<f8",
    ) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, dtype, 0))
        self.prefix = prefix
        self.suffix = ".dat" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class ElementalNpy(NdarrayFile, Elemental):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int] = [4, 123, 128, 70, 70],
        totNe: int = 70,
    ) -> None:
        super().__init__()
        Elemental.__init__(self, FileMetaData(shape, "<c8", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.meson.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class Jpsi2gammaNpy(NdarrayFile, TwoPoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        TwoPoint.__init__(self, None)
        self.prefix = prefix
        self.suffix = ".2pt.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class OnePointNpy(NdarrayFile, OnePoint):
    def __init__(self, prefix: str, suffix: str) -> None:
        super().__init__()
        # [2, 123, 128]
        OnePoint.__init__(self, None)
        self.prefix = prefix
        self.suffix = ".1pt.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class CurrentElementalV2P(NdarrayFile, CurrentElemental):
    """
    Current elemental v2p (eigenvector to point) data loader.

    Shape: [Lt, num_disp, Ne, Np, Nc]
    Single file format: {key}_v2p.npy
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],  # [Lt, num_disp, Ne, Np, Nc]
        Ne: int,
        Np: int,
    ) -> None:
        super().__init__()
        CurrentElemental.__init__(self, FileMetaData(shape, "<c16", 2), Ne, Np)
        self.prefix = prefix
        self.suffix = "_v2p.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class CurrentElementalP2V(NdarrayFile, CurrentElemental):
    """
    Current elemental p2v (point to eigenvector) data loader.

    Shape: [Lt, num_disp, Np, Nc, Ne]
    Single file format: {key}_p2v.npy
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],  # [Lt, num_disp, Np, Nc, Ne]
        Ne: int,
        Np: int,
    ) -> None:
        super().__init__()
        CurrentElemental.__init__(self, FileMetaData(shape, "<c16", 2), Ne, Np)
        self.prefix = prefix
        self.suffix = "_p2v.npy" if suffix is None else suffix

    def load(self, key: str):
        return super().get_file_data(f"{self.prefix}{key}{self.suffix}", self.elem)


class CurrentElementalP2P:
    """
    Current elemental p2p (point to point) sparse data loader.

    Supports both HDF5 and numpy.savez formats.
    - HDF5 format: {key}_p2p.h5 with structure disp_{idx}/t_{t}/type, indices, values
    - npz format: .t{t:03d}.p2p.npz with structure type_{i}, indices_{i}, values_{i}

    Note: p2p is momentum-independent, so each disp_idx data is repeated for all momentum_idx.

    Data structure per (disp_idx, momentum_idx):
        - type='identity': no data, represents delta_{l,r} delta_{c,c'}
        - type='sparse': indices [N,2], values [N,3,3]
    """

    def __init__(self, prefix: str, suffix: str = None) -> None:
        self.prefix = prefix
        # Default to HDF5 format if suffix is None
        self.suffix = suffix  # If None, will try HDF5 format first
        self.file = None
        self.data = None

    def load(self, key: str, t: int, num_momentum: int = None):
        """
        Load p2p data for specific configuration and time slice.

        Args:
            key: Configuration key (e.g., '10000')
            t: Time slice index
            num_momentum: Number of momentum indices (required for HDF5 format).
                         If None, will try to infer from file or use npz format.

        Returns:
            List of dicts, one per (disp_idx, momentum_idx).
            Length = num_disp * num_momentum.
            Since p2p is momentum-independent, same disp_idx data is repeated for all momentum_idx.
        """
        import numpy
        import re
        import h5py
        import os

        # Try HDF5 format first (if suffix is None or ends with .h5)
        h5_filename = f"{self.prefix}{key}_p2p.h5"
        if (self.suffix is None or self.suffix.endswith(".h5")) and os.path.exists(
            h5_filename
        ):
            return self._load_hdf5(h5_filename, t, num_momentum)

        # Fall back to npz format
        if self.suffix is None:
            suffix = ".t???.p2p.npz"
        else:
            suffix = self.suffix

        filename = f"{self.prefix}{key}{suffix}"
        filename = re.sub(r"\.t\?\?\?\.", f".t{t:03d}.", filename)

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"P2P data file not found: {filename} or {h5_filename}"
            )

        data = numpy.load(filename, allow_pickle=True)

        # Reconstruct list of dicts
        result = []
        num_entries = len([k for k in data.files if k.startswith("type_")])

        for i in range(num_entries):
            entry_type = str(data[f"type_{i}"])

            if entry_type == "identity":
                result.append({"type": "identity"})
            else:
                indices = data[f"indices_{i}"]
                values = data[f"values_{i}"]
                result.append(
                    {
                        "type": "sparse",
                        "indices": indices,
                        "values": values,
                    }
                )

        # If num_momentum is provided and result length is num_disp (not num_disp * num_momentum),
        # repeat each disp_idx entry for all momentum_idx
        if num_momentum is not None and len(result) > 0:
            num_disp = len(result)
            if num_disp < num_disp * num_momentum:
                # Repeat each disp_idx entry for all momentum_idx
                expanded_result = []
                for disp_idx in range(num_disp):
                    for momentum_idx in range(num_momentum):
                        expanded_result.append(result[disp_idx])
                result = expanded_result

        self.file = filename
        self.data = result
        return result

    def _load_hdf5(self, h5_filename: str, t: int, num_momentum: int):
        """
        Load p2p data from HDF5 format.

        Args:
            h5_filename: HDF5 file path
            t: Time slice index
            num_momentum: Number of momentum indices

        Returns:
            List of dicts, one per (disp_idx, momentum_idx)
        """
        import h5py
        import numpy

        if num_momentum is None:
            raise ValueError("num_momentum must be provided for HDF5 format")

        result = []

        with h5py.File(h5_filename, "r") as f:
            # Get number of displacements from file structure
            disp_indices = sorted(
                [int(k.split("_")[1]) for k in f.keys() if k.startswith("disp_")]
            )

            for disp_idx in disp_indices:
                disp_group = f[f"disp_{disp_idx}"]
                t_group = disp_group[f"t_{t}"]

                entry_type = (
                    t_group.attrs["type"].decode()
                    if isinstance(t_group.attrs["type"], bytes)
                    else str(t_group.attrs["type"])
                )

                if entry_type == "identity":
                    # Repeat identity for all momentum_idx
                    for momentum_idx in range(num_momentum):
                        result.append({"type": "identity"})
                else:
                    # Load sparse data
                    indices = numpy.array(t_group["indices"])
                    values = numpy.array(t_group["values"])

                    # Repeat sparse data for all momentum_idx
                    for momentum_idx in range(num_momentum):
                        result.append(
                            {
                                "type": "sparse",
                                "indices": indices,
                                "values": values,
                            }
                        )

        self.file = h5_filename
        self.data = result
        return result
