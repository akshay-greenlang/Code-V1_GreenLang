"""
PFDCalculator - Probability of Failure on Demand Calculator

This module implements PFD calculations per IEC 61511 for Safety Instrumented
Systems (SIS). It supports various voting architectures and provides
deterministic, auditable calculations.

Key formulas implemented:
- PFDavg for 1oo1, 1oo2, 2oo2, 2oo3 architectures
- Common Cause Failure (CCF) adjustments
- Proof test interval effects
- Dangerous detected vs undetected failure rates

Reference: IEC 61511-1, IEC 61508-6

Example:
    >>> from greenlang.safety.sil.pfd_calculator import PFDCalculator, PFDInput
    >>> calc = PFDCalculator()
    >>> input_data = PFDInput(
    ...     architecture="1oo2",
    ...     lambda_du=1e-6,  # Dangerous undetected failure rate
    ...     proof_test_interval_hours=8760,
    ...     beta_ccf=0.1
    ... )
    >>> result = calc.calculate(input_data)
    >>> print(f"PFDavg: {result.pfd_avg:.2e}")
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from decimal import Decimal
import hashlib
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class VotingArchitecture(str, Enum):
    """Supported voting architectures per IEC 61511."""

    ONE_OO_ONE = "1oo1"  # Single channel
    ONE_OO_TWO = "1oo2"  # Dual redundant (1 out of 2)
    TWO_OO_TWO = "2oo2"  # Dual series (2 out of 2)
    TWO_OO_THREE = "2oo3"  # Triple redundant (2 out of 3)
    ONE_OO_ONE_D = "1oo1D"  # Single with diagnostics
    ONE_OO_TWO_D = "1oo2D"  # Dual with diagnostics
    TWO_OO_THREE_D = "2oo3D"  # Triple with diagnostics


class FailureRateData(BaseModel):
    """Failure rate data for a component."""

    lambda_d: float = Field(
        ...,
        ge=0,
        description="Total dangerous failure rate (per hour)"
    )
    lambda_du: float = Field(
        ...,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    lambda_dd: float = Field(
        default=0,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    lambda_s: float = Field(
        default=0,
        ge=0,
        description="Safe failure rate (per hour)"
    )

    @model_validator(mode='after')
    def validate_failure_rates(self) -> 'FailureRateData':
        """Validate failure rate consistency."""
        if self.lambda_dd == 0:
            # If lambda_dd not provided, calculate from lambda_d - lambda_du
            self.lambda_dd = max(0, self.lambda_d - self.lambda_du)
        return self


class PFDInput(BaseModel):
    """Input parameters for PFD calculation."""

    architecture: VotingArchitecture = Field(
        ...,
        description="Voting architecture (e.g., 1oo2)"
    )
    lambda_du: float = Field(
        ...,
        ge=0,
        description="Dangerous undetected failure rate (per hour)"
    )
    lambda_dd: float = Field(
        default=0,
        ge=0,
        description="Dangerous detected failure rate (per hour)"
    )
    proof_test_interval_hours: float = Field(
        ...,
        gt=0,
        description="Proof test interval (hours)"
    )
    diagnostic_test_interval_hours: float = Field(
        default=1.0,
        gt=0,
        description="Diagnostic test interval (hours)"
    )
    mean_time_to_repair_hours: float = Field(
        default=8.0,
        ge=0,
        description="Mean time to repair (hours)"
    )
    beta_ccf: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Common cause failure beta factor"
    )
    diagnostic_coverage: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Diagnostic coverage (DC) factor"
    )
    mission_time_hours: Optional[float] = Field(
        None,
        gt=0,
        description="Mission time (hours). If None, uses proof test interval."
    )
    component_id: Optional[str] = Field(
        None,
        description="Component identifier for traceability"
    )

    @field_validator('beta_ccf')
    @classmethod
    def validate_beta(cls, v: float) -> float:
        """Validate CCF beta factor is realistic."""
        if v < 0.01:
            logger.warning(
                f"Beta CCF {v} is very low. Typical values are 0.05-0.2"
            )
        if v > 0.2:
            logger.warning(
                f"Beta CCF {v} is high. Consider improving diversity."
            )
        return v


class PFDResult(BaseModel):
    """Result of PFD calculation."""

    architecture: VotingArchitecture = Field(
        ...,
        description="Voting architecture used"
    )
    pfd_avg: float = Field(
        ...,
        ge=0,
        le=1,
        description="Average PFD over proof test interval"
    )
    pfd_max: float = Field(
        ...,
        ge=0,
        le=1,
        description="Maximum PFD (at end of proof test interval)"
    )
    sil_achieved: int = Field(
        ...,
        ge=0,
        le=4,
        description="SIL level achieved (0 if below SIL 1)"
    )
    lambda_du_effective: float = Field(
        ...,
        description="Effective dangerous undetected failure rate"
    )
    ccf_contribution: float = Field(
        ...,
        description="PFD contribution from common cause failures"
    )
    independent_contribution: float = Field(
        ...,
        description="PFD contribution from independent failures"
    )
    proof_test_interval_hours: float = Field(
        ...,
        description="Proof test interval used"
    )
    risk_reduction_factor: float = Field(
        ...,
        description="Risk Reduction Factor (1/PFD)"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of calculation"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    formula_used: str = Field(
        ...,
        description="Formula used for calculation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Calculation warnings"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PFDCalculator:
    """
    Probability of Failure on Demand Calculator.

    Implements PFD calculations per IEC 61511/61508 for various
    voting architectures. All calculations are deterministic with
    complete audit trail.

    The calculator supports:
    - 1oo1, 1oo2, 2oo2, 2oo3 architectures
    - Common cause failure adjustments
    - Diagnostic coverage effects
    - Proof test interval optimization

    Attributes:
        sil_pfd_ranges: Dict mapping SIL levels to PFD ranges

    Example:
        >>> calc = PFDCalculator()
        >>> result = calc.calculate(pfd_input)
        >>> print(f"SIL Achieved: SIL {result.sil_achieved}")
    """

    # SIL PFD ranges per IEC 61511 (low demand mode)
    SIL_PFD_RANGES: Dict[int, Tuple[float, float]] = {
        4: (1e-5, 1e-4),
        3: (1e-4, 1e-3),
        2: (1e-3, 1e-2),
        1: (1e-2, 1e-1),
        0: (1e-1, 1.0),  # Below SIL 1
    }

    def __init__(self):
        """Initialize PFDCalculator."""
        logger.info("PFDCalculator initialized")

    def calculate(self, input_data: PFDInput) -> PFDResult:
        """
        Calculate PFD for given input parameters.

        Args:
            input_data: PFDInput with all required parameters

        Returns:
            PFDResult with calculated PFD and related metrics

        Raises:
            ValueError: If input parameters are invalid
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []

        logger.info(
            f"Calculating PFD for {input_data.architecture.value} architecture"
        )

        try:
            # Get appropriate calculation method
            calc_method = self._get_calculation_method(input_data.architecture)

            # Calculate PFD
            pfd_result = calc_method(input_data)
            pfd_avg = pfd_result['pfd_avg']
            pfd_max = pfd_result['pfd_max']
            ccf_contrib = pfd_result['ccf_contribution']
            ind_contrib = pfd_result['independent_contribution']
            formula = pfd_result['formula']

            # Validate result
            if pfd_avg > 1.0:
                warnings.append(
                    f"Calculated PFD {pfd_avg} exceeds 1.0. "
                    "Check input parameters."
                )
                pfd_avg = min(pfd_avg, 1.0)

            # Determine SIL achieved
            sil_achieved = self._pfd_to_sil(pfd_avg)

            # Calculate Risk Reduction Factor
            rrf = 1.0 / pfd_avg if pfd_avg > 0 else float('inf')

            # Check for low demand assumption
            ti = input_data.proof_test_interval_hours
            if input_data.lambda_du * ti > 0.1:
                warnings.append(
                    "Lambda_DU * TI > 0.1. Low demand approximation may not be valid."
                )

            # Generate provenance hash
            provenance_hash = self._calculate_provenance(input_data, pfd_avg)

            result = PFDResult(
                architecture=input_data.architecture,
                pfd_avg=pfd_avg,
                pfd_max=pfd_max,
                sil_achieved=sil_achieved,
                lambda_du_effective=input_data.lambda_du,
                ccf_contribution=ccf_contrib,
                independent_contribution=ind_contrib,
                proof_test_interval_hours=input_data.proof_test_interval_hours,
                risk_reduction_factor=rrf,
                calculation_timestamp=start_time,
                provenance_hash=provenance_hash,
                formula_used=formula,
                warnings=warnings,
            )

            logger.info(
                f"PFD calculation complete. PFDavg: {pfd_avg:.2e}, "
                f"SIL: {sil_achieved}"
            )

            return result

        except Exception as e:
            logger.error(f"PFD calculation failed: {str(e)}", exc_info=True)
            raise

    def _get_calculation_method(self, architecture: VotingArchitecture):
        """Get the appropriate calculation method for architecture."""
        methods = {
            VotingArchitecture.ONE_OO_ONE: self._calc_1oo1,
            VotingArchitecture.ONE_OO_TWO: self._calc_1oo2,
            VotingArchitecture.TWO_OO_TWO: self._calc_2oo2,
            VotingArchitecture.TWO_OO_THREE: self._calc_2oo3,
            VotingArchitecture.ONE_OO_ONE_D: self._calc_1oo1d,
            VotingArchitecture.ONE_OO_TWO_D: self._calc_1oo2d,
            VotingArchitecture.TWO_OO_THREE_D: self._calc_2oo3d,
        }
        return methods[architecture]

    def _calc_1oo1(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 1oo1 (single channel) architecture.

        Formula: PFDavg = lambda_DU * TI / 2

        Per IEC 61508-6 B.3.2.2
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du

        # PFDavg for 1oo1
        pfd_avg = lambda_du * ti / 2.0

        # PFDmax at end of proof test interval
        pfd_max = lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,  # No CCF for single channel
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU * TI / 2'
        }

    def _calc_1oo2(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 1oo2 (dual redundant) architecture.

        Formula: PFDavg = ((1-beta) * lambda_DU)^2 * TI^2 / 3 + beta * lambda_DU * TI / 2

        Per IEC 61508-6 B.3.2.4
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        beta = input_data.beta_ccf

        # Independent failure contribution
        ind_contrib = ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_contrib + ccf_contrib

        # PFDmax
        pfd_max = ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_contrib,
            'formula': 'PFDavg = ((1-beta)*lambda_DU)^2*TI^2/3 + beta*lambda_DU*TI/2'
        }

    def _calc_2oo2(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 2oo2 (dual series) architecture.

        Formula: PFDavg = lambda_DU * TI (approximately 2 * single channel)

        Note: 2oo2 increases availability but decreases safety (higher PFD).
        Per IEC 61508-6 B.3.2.3
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du

        # For 2oo2, both channels must fail for safe action
        # This is actually less safe than 1oo1
        pfd_avg = 2.0 * lambda_du * ti / 2.0  # Simplified approximation

        pfd_max = 2.0 * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU * TI (2oo2 increases availability, not safety)'
        }

    def _calc_2oo3(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 2oo3 (triple modular redundant) architecture.

        Formula: PFDavg = 6 * ((1-beta)*lambda_DU)^2 * TI^2 / 3 + beta*lambda_DU*TI/2

        Per IEC 61508-6 B.3.2.6
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        beta = input_data.beta_ccf

        # Independent failure contribution (need 2 of 3 to fail)
        # C(3,2) = 3 combinations of 2 failures from 3 channels
        ind_contrib = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_contrib + ccf_contrib

        # PFDmax
        pfd_max = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_contrib,
            'formula': 'PFDavg = 3*((1-beta)*lambda_DU)^2*TI^2/3 + beta*lambda_DU*TI/2'
        }

    def _calc_1oo1d(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 1oo1D (single channel with diagnostics).

        Includes effect of diagnostic testing and MTTR.

        Formula: PFDavg = lambda_DU*TI/2 + lambda_DD*MTTR

        Per IEC 61508-6 B.3.2.2
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        mttr = input_data.mean_time_to_repair_hours

        # Undetected failures (tested at proof test)
        pfd_du = lambda_du * ti / 2.0

        # Detected failures (repaired after detection)
        pfd_dd = lambda_dd * mttr

        pfd_avg = pfd_du + pfd_dd
        pfd_max = lambda_du * ti + lambda_dd * mttr

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': 0.0,
            'independent_contribution': pfd_avg,
            'formula': 'PFDavg = lambda_DU*TI/2 + lambda_DD*MTTR'
        }

    def _calc_1oo2d(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 1oo2D architecture with diagnostics.

        Per IEC 61508-6 B.3.2.4 with diagnostic coverage
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        beta = input_data.beta_ccf
        mttr = input_data.mean_time_to_repair_hours

        # Independent DU failures
        ind_du = ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # Independent DD failures
        ind_dd = ((1 - beta) * lambda_dd) ** 2 * mttr ** 2

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_du + ind_dd + ccf_contrib

        pfd_max = ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_du + ind_dd,
            'formula': 'PFDavg = ((1-beta)*lambda_DU)^2*TI^2/3 + ((1-beta)*lambda_DD)^2*MTTR^2 + beta*lambda_DU*TI/2'
        }

    def _calc_2oo3d(self, input_data: PFDInput) -> Dict[str, Any]:
        """
        Calculate PFD for 2oo3D architecture with diagnostics.

        Per IEC 61508-6 B.3.2.6 with diagnostic coverage
        """
        ti = input_data.proof_test_interval_hours
        lambda_du = input_data.lambda_du
        lambda_dd = input_data.lambda_dd
        beta = input_data.beta_ccf
        mttr = input_data.mean_time_to_repair_hours

        # Independent DU failures (2 of 3 must fail)
        ind_du = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 / 3.0

        # Independent DD failures
        ind_dd = 3 * ((1 - beta) * lambda_dd) ** 2 * mttr ** 2

        # CCF contribution
        ccf_contrib = beta * lambda_du * ti / 2.0

        pfd_avg = ind_du + ind_dd + ccf_contrib

        pfd_max = 3 * ((1 - beta) * lambda_du) ** 2 * ti ** 2 + beta * lambda_du * ti

        return {
            'pfd_avg': pfd_avg,
            'pfd_max': pfd_max,
            'ccf_contribution': ccf_contrib,
            'independent_contribution': ind_du + ind_dd,
            'formula': 'PFDavg = 3*((1-beta)*lambda_DU)^2*TI^2/3 + 3*((1-beta)*lambda_DD)^2*MTTR^2 + beta*lambda_DU*TI/2'
        }

    def _pfd_to_sil(self, pfd: float) -> int:
        """
        Convert PFD to SIL level.

        Args:
            pfd: Probability of Failure on Demand

        Returns:
            SIL level (0-4)
        """
        for sil, (lower, upper) in self.SIL_PFD_RANGES.items():
            if lower <= pfd < upper:
                return sil

        if pfd < 1e-5:
            return 4  # Better than SIL 4

        return 0  # Below SIL 1

    def _calculate_provenance(
        self,
        input_data: PFDInput,
        pfd_avg: float
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_str = (
            f"{input_data.architecture.value}|"
            f"{input_data.lambda_du}|"
            f"{input_data.proof_test_interval_hours}|"
            f"{input_data.beta_ccf}|"
            f"{pfd_avg}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def optimize_proof_test_interval(
        self,
        target_pfd: float,
        lambda_du: float,
        architecture: VotingArchitecture,
        beta_ccf: float = 0.1,
        min_interval_hours: float = 720,  # 1 month
        max_interval_hours: float = 87600,  # 10 years
    ) -> float:
        """
        Find optimal proof test interval to achieve target PFD.

        Uses binary search to find the maximum interval that
        achieves the target PFD.

        Args:
            target_pfd: Target PFDavg to achieve
            lambda_du: Dangerous undetected failure rate
            architecture: Voting architecture
            beta_ccf: Common cause failure factor
            min_interval_hours: Minimum proof test interval
            max_interval_hours: Maximum proof test interval

        Returns:
            Optimal proof test interval in hours
        """
        logger.info(
            f"Optimizing proof test interval for target PFD {target_pfd:.2e}"
        )

        low = min_interval_hours
        high = max_interval_hours
        optimal = min_interval_hours

        while low <= high:
            mid = (low + high) / 2

            input_data = PFDInput(
                architecture=architecture,
                lambda_du=lambda_du,
                proof_test_interval_hours=mid,
                beta_ccf=beta_ccf
            )

            result = self.calculate(input_data)

            if result.pfd_avg <= target_pfd:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1

        logger.info(f"Optimal proof test interval: {optimal} hours")
        return optimal

    def compare_architectures(
        self,
        lambda_du: float,
        proof_test_interval_hours: float,
        beta_ccf: float = 0.1
    ) -> Dict[VotingArchitecture, PFDResult]:
        """
        Compare PFD across all voting architectures.

        Args:
            lambda_du: Dangerous undetected failure rate
            proof_test_interval_hours: Proof test interval
            beta_ccf: Common cause failure factor

        Returns:
            Dict mapping architecture to PFDResult
        """
        results = {}

        for arch in VotingArchitecture:
            try:
                input_data = PFDInput(
                    architecture=arch,
                    lambda_du=lambda_du,
                    proof_test_interval_hours=proof_test_interval_hours,
                    beta_ccf=beta_ccf
                )
                results[arch] = self.calculate(input_data)
            except Exception as e:
                logger.warning(f"Failed to calculate PFD for {arch}: {e}")

        return results
