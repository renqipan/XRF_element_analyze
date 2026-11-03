from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import math


@dataclass
class AnalysisConfig:
    energy_range: Optional[Tuple[float, float]] = None
    find_peaks: Optional[Tuple[float, ...]] = None
    find_peaks_axis: str = "reference"
    manual_match_tolerance_kev: float = 0.08
    element_prior: str = "general"
    prior_tie_margin: float = 0.5
    branch_evidence: bool = True
    low_z_kalpha: bool = True
    low_z_max_z: int = 24
    low_z_kalpha_min_delta: float = 25.0
    low_z_beta_visibility_sigma: float = 5.0
    low_z_min_kalpha_share: float = 0.7
    prune_unanchored_auxiliary: bool = True
    complete_atomic_lines: bool = True
    shell_consistency_tolerance: float = 0.05
    minimum_radiative_rate: float = 0.001
    infer_low_energy_cutoff: bool = True
    alignment_max_shift_kev: float = 0.25
    centroid_tolerance_kev: float = 0.015
    pattern_element_penalty: float = 4.0
    channel_start: int = 0
    fill_missing_channels: bool = False
    strict_calibration: bool = False
    fwhm_ref_energy_kev: float = 5.9
    fwhm_ref_kev: float = 0.135
    fano: float = 0.115
    pair_energy_kev: float = 0.0036
    fit_response: bool = True
    estimate_resolution_from_peaks: bool = True
    fit_energy_gain: bool = True
    energy_gain_bound: float = 0.01
    resolution_scale_bounds: Tuple[float, float] = (0.75, 1.5)
    delta_max_kev: Optional[float] = None
    snip_iterations: Optional[int] = None
    background_degree: int = 3
    background_regularization: float = 0.1
    candidate_threshold_sigma: float = 3.5
    max_deviance_per_dof: float = 5.0
    support_delta_deviance: float = 25.0
    tentative_delta_deviance: float = 6.0
    match_sigma: float = 2.5
    correlation_threshold: float = 0.995
    elements: Optional[Tuple[str, ...]] = None
    exclude_elements: Tuple[str, ...] = ()
    element_selection: str = "model-comparison"
    max_selection_iterations: int = 3
    reference_element: Optional[str] = None
    tube_target: Optional[str] = None
    tube_voltage_kv: Optional[float] = None
    scatter_angle_deg: Optional[float] = None
    detector_efficiency: float = 1.0
    bootstrap: int = 0
    seed: int = 42
    max_iterations: int = 1800
    save_intermediate: bool = False

    def validate(self):
        from .element_prior import PROFILES
        if self.element_prior not in PROFILES:
            raise ValueError("Unknown element_prior profile")
        if not math.isfinite(self.prior_tie_margin) or not 0 <= self.prior_tie_margin <= 2:
            raise ValueError("prior_tie_margin must be finite and in [0,2]")
        if self.find_peaks_axis not in ("reference", "observed"):
            raise ValueError("find_peaks_axis must be reference or observed")
        if self.manual_match_tolerance_kev <= 0 or not math.isfinite(self.manual_match_tolerance_kev):
            raise ValueError("manual_match_tolerance_kev must be finite and positive")
        if self.find_peaks is not None:
            if isinstance(self.find_peaks, (str, bytes)) or not len(self.find_peaks):
                raise ValueError("find_peaks must be None or a nonempty list of keV energies")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in self.find_peaks):
                raise ValueError("find_peaks energies must be positive finite numbers in keV")
            if len(set(self.find_peaks)) != len(self.find_peaks):
                raise ValueError("Exact duplicate find_peaks centers are not allowed; close unresolved centers are supported")
        if not 1 <= self.low_z_max_z <= 118:
            raise ValueError("low_z_max_z must be in 1..118")
        if not 0.5 < self.low_z_min_kalpha_share <= 1:
            raise ValueError("low_z_min_kalpha_share must be in (0.5,1]")
        if not all(math.isfinite(v) and v > 0 for v in (self.low_z_kalpha_min_delta, self.low_z_beta_visibility_sigma)):
            raise ValueError("Low-Z evidence thresholds must be finite and positive")
        numeric = (self.fwhm_ref_energy_kev, self.fwhm_ref_kev, self.fano,
                   self.pair_energy_kev, self.candidate_threshold_sigma,
                   self.support_delta_deviance, self.tentative_delta_deviance,
                   self.background_regularization, self.match_sigma)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Configuration values must be finite")
        if not math.isfinite(self.max_deviance_per_dof) or self.max_deviance_per_dof <= 0:
            raise ValueError("max_deviance_per_dof must be finite and >0")
        if not all(math.isfinite(v) and v > 0 for v in (
                self.shell_consistency_tolerance, self.minimum_radiative_rate,
                self.alignment_max_shift_kev, self.centroid_tolerance_kev, self.pattern_element_penalty, self.energy_gain_bound)):
            raise ValueError("Invalid atomic completion or centroid matching parameters")
        if self.channel_start not in (0, 1):
            raise ValueError("channel_start must be 0 or 1")
        if self.energy_range is not None and (not all(math.isfinite(x) for x in self.energy_range)
                or self.energy_range[0] < 0 or self.energy_range[1] <= self.energy_range[0]):
            raise ValueError("energy_range must satisfy 0 <= min < max")
        if self.detector_efficiency != 1.0:
            raise ValueError("MVP requires detector_efficiency=1.0; S already includes its efficiency convention")
        if not 0 <= self.background_degree <= 5 or self.bootstrap < 0:
            raise ValueError("background_degree must be 0..5 and bootstrap >= 0")
        if self.fano < 0 or self.pair_energy_kev <= 0 or self.fwhm_ref_energy_kev <= 0:
            raise ValueError("Invalid resolution parameters")
        if self.fwhm_ref_kev <= 0 or (self.fwhm_ref_kev / 2.354820045) ** 2 < self.fano * self.pair_energy_kev * self.fwhm_ref_energy_kev:
            raise ValueError("Reference FWHM is below the configured Fano limit")
        if self.snip_iterations is not None and self.snip_iterations < 1:
            raise ValueError("snip_iterations must be positive")
        if self.background_regularization < 0 or self.candidate_threshold_sigma <= 0 or self.match_sigma <= 0:
            raise ValueError("Invalid background/detection parameters")
        if not 0 < self.tentative_delta_deviance < self.support_delta_deviance:
            raise ValueError("Require 0 < tentative threshold < support threshold")
        if not 0 < self.correlation_threshold < 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.delta_max_kev is not None and (not math.isfinite(self.delta_max_kev) or self.delta_max_kev <= 0):
            raise ValueError("delta_max_kev must be finite and positive")
        if self.tube_voltage_kv is not None and (not math.isfinite(self.tube_voltage_kv) or self.tube_voltage_kv <= 0):
            raise ValueError("tube_voltage_kv must be finite and positive")
        if self.scatter_angle_deg is not None and (not math.isfinite(self.scatter_angle_deg) or not 0 <= self.scatter_angle_deg <= 180):
            raise ValueError("scatter_angle_deg must be 0..180")
        lo, hi = self.resolution_scale_bounds
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
            raise ValueError("Invalid resolution scale bounds")
        if self.max_selection_iterations < 1 or self.max_iterations < 1:
            raise ValueError("Iteration limits must be positive")
        if self.element_selection not in ("manual", "model-comparison"):
            raise ValueError("Unknown element selection method")
        if self.element_selection == "manual" and not self.elements:
            raise ValueError("manual selection requires an explicit elements whitelist")
        return self

    def to_dict(self):
        return asdict(self)
