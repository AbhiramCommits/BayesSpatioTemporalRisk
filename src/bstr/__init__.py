"""BayesSpatioTemporalRisk: Bayesian spatio-temporal point-process risk modelling.

Modules
-------
config    : frozen simulation/prior configuration objects and project paths
grid      : square-cell geometry and rook contiguity
simulate  : seeded ground-truth simulator for raw event and mileage logs
aggregate : PySpark/SQL rollup of raw logs to a cell x time-block exposure table
baselines : frequentist pooled / fixed-effects / negative-binomial Poisson GLMs
model     : PyMC hierarchical ICAR + AR(1) Poisson model with exposure offset
checks    : posterior predictive checks, LOO/WAIC comparison, prior sensitivity
readout   : posterior assurance decisions and the stakeholder memo
"""

__version__ = "0.1.0"


def _patch_pytensor_macos_linker() -> None:
    """Work around an invalid PyTensor 2.31 linker flag on recent macOS.

    ``GCC_compiler.compile_args`` appends ``-ld64`` as a *compile* flag when the
    macOS major version is >= 15, which clang interprets as ``-l d64`` and fails
    to link. The intended linker flag is ``-Wl,-ld64``. Patch it before PyMC
    compiles anything; harmless elsewhere.
    """
    import platform
    import sys

    if sys.platform != "darwin":
        return
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        return
    if major < 15:
        return

    from pytensor.link.c.cmodule import GCC_compiler

    original = GCC_compiler.compile_args
    if getattr(original, "_bstr_patched", False):
        return

    def compile_args(*args, **kwargs):
        return [f for f in original(*args, **kwargs) if f != "-ld64"]

    compile_args._bstr_patched = True
    GCC_compiler.compile_args = staticmethod(compile_args)


_patch_pytensor_macos_linker()
