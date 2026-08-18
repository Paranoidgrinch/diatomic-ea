"""def2 orbital-basis and ECP resolution.

The resolver follows the established DiatomicEA research workflow:

1. Prefer PySCF's built-in basis library.
2. Use all-electron def2 calculations through Kr.
3. For heavier elements, search compatible def2 ECP names.
4. Fall back to Basis Set Exchange when necessary.

PySCF and Basis Set Exchange are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

from diatomic_ea.molecule import normalize_element_symbol


ALL_ELECTRON_DEF2_ELEMENTS = frozenset(
    {
        "H", "He",
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
        "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
        "Br", "Kr",
    }
)


@dataclass(frozen=True, slots=True)
class BasisECPResolution:
    """Resolved orbital basis and optional ECP for one element."""

    element: str
    requested_basis: str
    basis_spec: object
    ecp_spec: object
    basis_label: str
    ecp_label: str

    @property
    def uses_ecp(self) -> bool:
        return bool(self.ecp_spec)


def uses_def2_ecp(element: str) -> bool:
    """Return whether a standard def2 calculation requires an ECP."""
    symbol = normalize_element_symbol(element)
    return symbol not in ALL_ELECTRON_DEF2_ELEMENTS


def ecp_candidates_for_basis(
    basis_name: str,
) -> tuple[str, ...]:
    """Return compatible PySCF ECP names in search order."""
    basis = basis_name.strip().lower()

    if not basis:
        raise ValueError(
            "basis_name must not be empty."
        )

    candidates: list[str] = [basis]

    if basis == "def2-svpd":
        candidates.append("def2-svp")

    if basis == "def2-tzvpd":
        candidates.append("def2-tzvp")

    if basis == "def2-qzvpd":
        candidates.extend(
            [
                "def2-qzvp",
                "def2-qzvpp",
            ]
        )

    if basis.endswith("vpd"):
        candidates.append(basis[:-1])

    candidates.append("def2-ecp")

    return tuple(
        dict.fromkeys(candidates)
    )


def bse_def2_name(
    basis_name: str,
) -> str:
    """Return the canonical Basis Set Exchange def2 name."""
    basis = basis_name.strip().lower()

    mapping = {
        "def2-svp": "def2-SVP",
        "def2-tzvp": "def2-TZVP",
        "def2-tzvpp": "def2-TZVPP",
        "def2-svpd": "def2-SVPD",
        "def2-tzvpd": "def2-TZVPD",
        "def2-qzvp": "def2-QZVP",
        "def2-qzvpp": "def2-QZVPP",
        "def2-qzvpd": "def2-QZVPD",
    }

    return mapping.get(
        basis,
        basis_name,
    )


def split_bse_nwchem_basis_ecp(
    text: str,
) -> tuple[str, str]:
    """Split combined BSE NWChem BASIS and ECP blocks."""
    basis_lines: list[str] = []
    ecp_lines: list[str] = []

    mode: str | None = None

    for line in text.splitlines():
        upper = line.strip().upper()

        if upper.startswith("BASIS"):
            mode = "basis"
            basis_lines.append(line)
            continue

        if upper == "ECP":
            mode = "ecp"
            ecp_lines.append(line)
            continue

        if mode == "basis":
            basis_lines.append(line)

            if upper == "END":
                mode = None

            continue

        if mode == "ecp":
            ecp_lines.append(line)

            if upper == "END":
                mode = None

    return (
        "\n".join(basis_lines).strip(),
        "\n".join(ecp_lines).strip(),
    )


class PySCFBasisResolver:
    """Resolve orbital bases and ECPs for PySCF."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, str],
            BasisECPResolution,
        ] = {}

    def resolve(
        self,
        element: str,
        basis_name: str,
    ) -> BasisECPResolution:
        """Resolve one element/basis combination."""
        symbol = normalize_element_symbol(element)
        basis = basis_name.strip().lower()

        if not basis:
            raise ValueError(
                "basis_name must not be empty."
            )

        key = (symbol, basis)

        if key in self._cache:
            return self._cache[key]

        from pyscf.gto import basis as basismod  # type: ignore

        basis_spec: object
        ecp_spec: object = ""
        basis_label = ""
        ecp_label = ""

        try:
            basismod.load(
                basis,
                symbol,
            )

            basis_spec = basis
            basis_label = f"pyscf:{basis}"

        except Exception as builtin_exc:
            try:
                resolution = self._load_from_bse(
                    symbol,
                    basis,
                )
            except Exception as bse_exc:
                raise RuntimeError(
                    f"Could not load orbital basis "
                    f"{basis_name!r} for {symbol}. "
                    f"PySCF error: {builtin_exc!r} | "
                    f"BSE fallback error: {bse_exc!r}"
                ) from bse_exc

            self._cache[key] = resolution
            return resolution

        if not uses_def2_ecp(symbol):
            resolution = BasisECPResolution(
                element=symbol,
                requested_basis=basis,
                basis_spec=basis_spec,
                ecp_spec="",
                basis_label=basis_label,
                ecp_label="",
            )

            self._cache[key] = resolution
            return resolution

        for candidate in ecp_candidates_for_basis(
            basis
        ):
            try:
                basismod.load_ecp(
                    candidate,
                    symbol,
                )

                ecp_spec = candidate
                ecp_label = f"pyscf:{candidate}"
                break

            except Exception:
                continue

        if not ecp_spec:
            try:
                bse_resolution = self._load_from_bse(
                    symbol,
                    basis,
                )

                if bse_resolution.ecp_spec:
                    ecp_spec = bse_resolution.ecp_spec
                    ecp_label = bse_resolution.ecp_label

            except Exception:
                pass

        resolution = BasisECPResolution(
            element=symbol,
            requested_basis=basis,
            basis_spec=basis_spec,
            ecp_spec=ecp_spec,
            basis_label=basis_label,
            ecp_label=ecp_label,
        )

        self._cache[key] = resolution
        return resolution

    def _load_from_bse(
        self,
        element: str,
        basis_name: str,
    ) -> BasisECPResolution:
        import basis_set_exchange as bse  # type: ignore
        from pyscf.gto.basis import (  # type: ignore
            parse_nwchem,
            parse_nwchem_ecp,
        )

        bse_name = bse_def2_name(
            basis_name
        )

        text = bse.get_basis(
            bse_name,
            elements=[element],
            fmt="nwchem",
        )

        basis_text, ecp_text = (
            split_bse_nwchem_basis_ecp(text)
        )

        if not basis_text:
            raise RuntimeError(
                f"BSE returned no BASIS block for "
                f"{element} / {bse_name}."
            )

        basis_spec = parse_nwchem.parse(
            basis_text,
            symb=element,
        )

        ecp_spec: object = ""
        ecp_label = ""

        if ecp_text:
            try:
                ecp_spec = parse_nwchem_ecp.parse(
                    ecp_text,
                    symb=element,
                )

                if ecp_spec:
                    ecp_label = f"bse:{bse_name}"

            except Exception as exc:
                ecp_spec = ""
                ecp_label = (
                    "bse_ecp_parse_failed:"
                    f"{exc!r}"
                )

        return BasisECPResolution(
            element=element,
            requested_basis=basis_name,
            basis_spec=basis_spec,
            ecp_spec=ecp_spec,
            basis_label=f"bse:{bse_name}",
            ecp_label=ecp_label,
        )