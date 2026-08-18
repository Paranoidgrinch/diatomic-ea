"""PySCF UKS single-point calculations for diatomic molecules."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from enum import Enum

from diatomic_ea.backend import PySCFBackend
from diatomic_ea.basis import PySCFBasisResolver
from diatomic_ea.geometry import DiatomicGeometry
from diatomic_ea.molecule import DiatomicMolecule
from diatomic_ea.schema_f import (
    HARTREE_TO_EV,
    SCHEMA_F,
)
from diatomic_ea.states import (
    ChargeState,
    ElectronicState,
    is_spin_contaminated,
)


class SinglePointStatus(str, Enum):
    """Execution status of one electronic-structure calculation."""

    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FrontierOrbitals:
    """Frontier-orbital diagnostics."""

    homo_hartree: float
    lumo_hartree: float
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    positive_homo_warning: bool


@dataclass(frozen=True, slots=True)
class SinglePointTask:
    """One UKS energy calculation."""

    molecule: DiatomicMolecule
    charge: ChargeState
    spin: int
    functional: str
    basis: str
    bond_length_angstrom: float
    grid_level: int
    conv_tol: float
    max_cycle: int
    max_memory_mb: int
    threads_per_worker: int = 1

    def __post_init__(self) -> None:
        ElectronicState(
            charge=self.charge,
            spin=self.spin,
        )

        if self.bond_length_angstrom <= 0:
            raise ValueError(
                "bond_length_angstrom must be positive."
            )

        if not self.functional.strip():
            raise ValueError(
                "functional must not be empty."
            )

        if not self.basis.strip():
            raise ValueError(
                "basis must not be empty."
            )

        if self.grid_level < 0:
            raise ValueError(
                "grid_level must not be negative."
            )

        if self.conv_tol <= 0:
            raise ValueError(
                "conv_tol must be positive."
            )

        if self.max_cycle < 1:
            raise ValueError(
                "max_cycle must be at least 1."
            )

        if self.max_memory_mb < 1:
            raise ValueError(
                "max_memory_mb must be at least 1."
            )

        if self.threads_per_worker < 1:
            raise ValueError(
                "threads_per_worker must be at least 1."
            )

    @property
    def multiplicity(self) -> int:
        return self.spin + 1

    @property
    def task_id(self) -> str:
        charge_tag = (
            "neutral"
            if self.charge is ChargeState.NEUTRAL
            else "anion"
        )

        return (
            f"{self.molecule.formula.lower()}_"
            f"{charge_tag}_spin{self.spin}_"
            f"{self.functional}_"
            f"{self.basis}_"
            f"r{format_float_for_id(self.bond_length_angstrom)}"
        )


@dataclass(frozen=True, slots=True)
class SinglePointResult:
    """Result and diagnostics from one UKS calculation."""

    task_id: str
    status: SinglePointStatus
    error: str
    energy_hartree: float
    energy_ev: float
    converged: bool
    used_level_shift_retry: bool
    used_newton_retry: bool
    electron_count: int | None
    alpha_electrons: int | None
    beta_electrons: int | None
    basis_label_a: str
    basis_label_b: str
    ecp_label_a: str
    ecp_label_b: str
    frontier: FrontierOrbitals | None
    s2: float
    observed_multiplicity: float
    spin_contamination_warning: bool
    pyscf_version: str
    elapsed_seconds: float


def functional_to_xc(name: str) -> str:
    """Map workflow functional labels to PySCF XC labels."""
    key = name.strip().upper()

    mapping = {
        "PBE": "PBE",
        "B3LYP": "B3LYP",
        "PBE0": "PBE0",
        "TPSH": "TPSSh",
        "TPSSH": "TPSSh",
    }

    return mapping.get(
        key,
        name.strip(),
    )


def format_float_for_id(value: float) -> str:
    """Create deterministic compact coordinates for task IDs."""
    return (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def compute_frontier_orbitals(
    mo_energy,
    mo_occ,
) -> FrontierOrbitals:
    """Calculate HOMO/LUMO diagnostics from UKS orbital data."""
    occupied: list[float] = []
    virtual: list[float] = []

    def collect(
        energies,
        occupations,
    ) -> None:
        for energy, occupation in zip(
            energies,
            occupations,
        ):
            value = float(energy)

            if float(occupation) > 1.0e-8:
                occupied.append(value)
            else:
                virtual.append(value)

    if isinstance(
        mo_energy,
        (list, tuple),
    ):
        for energies, occupations in zip(
            mo_energy,
            mo_occ,
        ):
            collect(
                energies,
                occupations,
            )
    else:
        collect(
            mo_energy,
            mo_occ,
        )

    homo = (
        max(occupied)
        if occupied
        else math.nan
    )

    lumo = (
        min(virtual)
        if virtual
        else math.nan
    )

    homo_ev = (
        homo * HARTREE_TO_EV
        if math.isfinite(homo)
        else math.nan
    )

    lumo_ev = (
        lumo * HARTREE_TO_EV
        if math.isfinite(lumo)
        else math.nan
    )

    gap_ev = (
        (lumo - homo) * HARTREE_TO_EV
        if (
            math.isfinite(homo)
            and math.isfinite(lumo)
        )
        else math.nan
    )

    return FrontierOrbitals(
        homo_hartree=homo,
        lumo_hartree=lumo,
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        gap_ev=gap_ev,
        positive_homo_warning=bool(
            math.isfinite(homo)
            and homo > 0.0
        ),
    )


def configure_worker_threads(
    threads_per_worker: int,
) -> None:
    """Prevent BLAS/OpenMP oversubscription in a worker."""
    value = str(
        int(threads_per_worker)
    )

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = value


def run_pyscf_single_point(
    task: SinglePointTask,
) -> SinglePointResult:
    """Execute one PySCF UKS calculation."""
    started = time.perf_counter()

    nan = math.nan

    backend = PySCFBackend()
    availability = backend.availability()

    if not availability.ready:
        return SinglePointResult(
            task_id=task.task_id,
            status=SinglePointStatus.ERROR,
            error=availability.message,
            energy_hartree=nan,
            energy_ev=nan,
            converged=False,
            used_level_shift_retry=False,
            used_newton_retry=False,
            electron_count=None,
            alpha_electrons=None,
            beta_electrons=None,
            basis_label_a="",
            basis_label_b="",
            ecp_label_a="",
            ecp_label_b="",
            frontier=None,
            s2=nan,
            observed_multiplicity=nan,
            spin_contamination_warning=False,
            pyscf_version="",
            elapsed_seconds=(
                time.perf_counter() - started
            ),
        )

    basis_label_a = ""
    basis_label_b = ""
    ecp_label_a = ""
    ecp_label_b = ""

    try:
        configure_worker_threads(
            task.threads_per_worker
        )

        from pyscf import (  # type: ignore
            __version__ as pyscf_version,
        )
        from pyscf import dft, gto, lib  # type: ignore

        lib.num_threads(
            task.threads_per_worker
        )

        resolver = PySCFBasisResolver()

        resolved_a = resolver.resolve(
            task.molecule.atom_a,
            task.basis,
        )

        resolved_b = resolver.resolve(
            task.molecule.atom_b,
            task.basis,
        )

        basis_label_a = resolved_a.basis_label
        basis_label_b = resolved_b.basis_label
        ecp_label_a = resolved_a.ecp_label
        ecp_label_b = resolved_b.ecp_label

        geometry = DiatomicGeometry(
            molecule=task.molecule,
            bond_length_angstrom=(
                task.bond_length_angstrom
            ),
        )

        mol = gto.Mole()
        mol.atom = geometry.atom_string
        mol.unit = "Angstrom"
        mol.charge = int(task.charge)
        mol.spin = task.spin

        mol.basis = {
            task.molecule.atom_a:
                resolved_a.basis_spec,
            task.molecule.atom_b:
                resolved_b.basis_spec,
        }

        ecp: dict[str, object] = {}

        if resolved_a.ecp_spec:
            ecp[
                task.molecule.atom_a
            ] = resolved_a.ecp_spec

        if resolved_b.ecp_spec:
            ecp[
                task.molecule.atom_b
            ] = resolved_b.ecp_spec

        if ecp:
            mol.ecp = ecp

        mol.max_memory = task.max_memory_mb
        mol.verbose = 0
        mol.symmetry = False
        mol.build()

        mf = dft.UKS(mol)
        mf.xc = functional_to_xc(
            task.functional
        )
        mf.conv_tol = task.conv_tol
        mf.max_cycle = task.max_cycle
        mf.grids.level = task.grid_level
        mf.max_memory = task.max_memory_mb
        mf.direct_scf = True
        mf.init_guess = "minao"
        mf.verbose = 0

        energy = mf.kernel()

        used_level_shift = False
        used_newton = False

        if not bool(
            getattr(mf, "converged", False)
        ):
            try:
                density = mf.make_rdm1()
                mf.level_shift = (
                    SCHEMA_F.scf_rescue.level_shift
                )

                energy = mf.kernel(
                    density
                )

                used_level_shift = True

            except Exception:
                pass

        if not bool(
            getattr(mf, "converged", False)
        ):
            try:
                density = mf.make_rdm1()

                newton_mf = mf.newton()
                newton_mf.max_cycle = (
                    SCHEMA_F.scf_rescue
                    .newton_max_cycle
                )
                newton_mf.conv_tol = task.conv_tol
                newton_mf.verbose = 0

                energy = newton_mf.kernel(
                    density
                )

                mf = newton_mf
                used_newton = True

            except Exception:
                pass

        frontier = compute_frontier_orbitals(
            mf.mo_energy,
            mf.mo_occ,
        )

        try:
            s2_raw, multiplicity_raw = (
                mf.spin_square()
            )

            s2 = float(s2_raw)
            observed_multiplicity = float(
                multiplicity_raw
            )

        except Exception:
            s2 = nan
            observed_multiplicity = nan

        contaminated = False

        if (
            math.isfinite(s2)
            and math.isfinite(
                observed_multiplicity
            )
        ):
            contaminated = (
                is_spin_contaminated(
                    input_spin=task.spin,
                    observed_s2=s2,
                    observed_multiplicity=(
                        observed_multiplicity
                    ),
                )
            )

        energy_float = float(energy)

        return SinglePointResult(
            task_id=task.task_id,
            status=SinglePointStatus.OK,
            error="",
            energy_hartree=energy_float,
            energy_ev=(
                energy_float
                * HARTREE_TO_EV
            ),
            converged=bool(
                getattr(
                    mf,
                    "converged",
                    False,
                )
            ),
            used_level_shift_retry=(
                used_level_shift
            ),
            used_newton_retry=(
                used_newton
            ),
            electron_count=int(
                mol.nelectron
            ),
            alpha_electrons=int(
                mol.nelec[0]
            ),
            beta_electrons=int(
                mol.nelec[1]
            ),
            basis_label_a=basis_label_a,
            basis_label_b=basis_label_b,
            ecp_label_a=ecp_label_a,
            ecp_label_b=ecp_label_b,
            frontier=frontier,
            s2=s2,
            observed_multiplicity=(
                observed_multiplicity
            ),
            spin_contamination_warning=(
                contaminated
            ),
            pyscf_version=str(
                pyscf_version
            ),
            elapsed_seconds=(
                time.perf_counter()
                - started
            ),
        )

    except Exception as exc:
        return SinglePointResult(
            task_id=task.task_id,
            status=SinglePointStatus.ERROR,
            error=repr(exc),
            energy_hartree=nan,
            energy_ev=nan,
            converged=False,
            used_level_shift_retry=False,
            used_newton_retry=False,
            electron_count=None,
            alpha_electrons=None,
            beta_electrons=None,
            basis_label_a=basis_label_a,
            basis_label_b=basis_label_b,
            ecp_label_a=ecp_label_a,
            ecp_label_b=ecp_label_b,
            frontier=None,
            s2=nan,
            observed_multiplicity=nan,
            spin_contamination_warning=False,
            pyscf_version=(
                availability.version or ""
            ),
            elapsed_seconds=(
                time.perf_counter()
                - started
            ),
        )


def determine_electron_count(
    molecule: DiatomicMolecule,
    *,
    charge: ChargeState,
    basis: str,
    bond_length_angstrom: float,
    max_memory_mb: int,
) -> int:
    """Probe PySCF for the explicit post-ECP electron count.

    Different spin guesses are tried because PySCF validates
    electron/spin parity during Mole.build().
    """
    errors: list[str] = []

    for spin in range(8):
        task = SinglePointTask(
            molecule=molecule,
            charge=charge,
            spin=spin,
            functional="PBE",
            basis=basis,
            bond_length_angstrom=bond_length_angstrom,
            grid_level=0,
            conv_tol=1.0e-8,
            max_cycle=1,
            max_memory_mb=max_memory_mb,
            threads_per_worker=1,
        )

        backend = PySCFBackend()

        if not backend.availability().ready:
            raise RuntimeError(
                backend.availability().message
            )

        try:
            from pyscf import gto  # type: ignore

            resolver = PySCFBasisResolver()

            resolved_a = resolver.resolve(
                molecule.atom_a,
                basis,
            )
            resolved_b = resolver.resolve(
                molecule.atom_b,
                basis,
            )

            geometry = DiatomicGeometry(
                molecule=molecule,
                bond_length_angstrom=(
                    bond_length_angstrom
                ),
            )

            mol = gto.Mole()
            mol.atom = geometry.atom_string
            mol.unit = "Angstrom"
            mol.charge = int(charge)
            mol.spin = task.spin
            mol.basis = {
                molecule.atom_a:
                    resolved_a.basis_spec,
                molecule.atom_b:
                    resolved_b.basis_spec,
            }

            ecp: dict[str, object] = {}

            if resolved_a.ecp_spec:
                ecp[molecule.atom_a] = (
                    resolved_a.ecp_spec
                )

            if resolved_b.ecp_spec:
                ecp[molecule.atom_b] = (
                    resolved_b.ecp_spec
                )

            if ecp:
                mol.ecp = ecp

            mol.max_memory = max_memory_mb
            mol.verbose = 0
            mol.symmetry = False
            mol.build()

            return int(
                mol.nelectron
            )

        except Exception as exc:
            errors.append(
                f"spin={spin}: {exc!r}"
            )

    raise RuntimeError(
        "Could not determine electron count for "
        f"{molecule.formula} charge={int(charge)}. "
        + " | ".join(errors)
    )