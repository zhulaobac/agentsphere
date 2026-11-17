"""Release Gate: mandatory check used by Control Plane before Approved → Released."""

from __future__ import annotations

from agentsphere.controlplane.types import GateCheck, GateDecision, GateThresholds
from agentsphere.evaluation.experiment import ExperimentRun
from agentsphere.evaluation.regression import compare_experiments


class ReleaseGate:
    """Offline quality / regression / safety / budget gate. No live production metrics required."""

    def evaluate(
        self,
        candidate: ExperimentRun,
        *,
        thresholds: GateThresholds | None = None,
        baseline: ExperimentRun | None = None,
    ) -> GateDecision:
        thresholds = thresholds or GateThresholds()
        checks: list[GateCheck] = []

        quality = candidate.summary.pass_rate
        checks.append(
            GateCheck(
                name="quality",
                passed=quality >= thresholds.min_quality,
                observed=quality,
                threshold=thresholds.min_quality,
                reason=f"pass_rate {quality:.3f} vs min {thresholds.min_quality:.3f}",
            )
        )

        if baseline is None:
            checks.append(
                GateCheck(
                    name="regression",
                    passed=True,
                    skipped=True,
                    reason="no baseline experiment",
                )
            )
        else:
            report = compare_experiments(baseline, candidate)
            drop = -report.delta if report.delta < 0 else 0.0
            checks.append(
                GateCheck(
                    name="regression",
                    passed=drop <= thresholds.max_regression,
                    observed=drop,
                    threshold=thresholds.max_regression,
                    reason=f"pass_rate drop {drop:.3f} vs max {thresholds.max_regression:.3f}",
                )
            )

        safety_cases = [c for c in candidate.cases if c.case.is_safety()]
        if not thresholds.require_safety_pass:
            checks.append(
                GateCheck(name="safety", passed=True, skipped=True, reason="safety check disabled")
            )
        elif not safety_cases:
            checks.append(
                GateCheck(
                    name="safety",
                    passed=True,
                    skipped=True,
                    reason="no safety-tagged cases",
                    observed=0,
                )
            )
        else:
            failed = [c.case.case_id for c in safety_cases if not c.pass_at_k]
            checks.append(
                GateCheck(
                    name="safety",
                    passed=not failed,
                    observed=len(safety_cases) - len(failed),
                    threshold=len(safety_cases),
                    reason="all safety cases passed" if not failed else f"safety failures: {failed}",
                )
            )

        if thresholds.max_cost is None:
            checks.append(GateCheck(name="cost", passed=True, skipped=True, reason="no cost budget"))
        elif candidate.summary.mean_cost is None:
            checks.append(
                GateCheck(
                    name="cost",
                    passed=True,
                    skipped=True,
                    reason="cost metric not present",
                )
            )
        else:
            checks.append(
                GateCheck(
                    name="cost",
                    passed=candidate.summary.mean_cost <= thresholds.max_cost,
                    observed=candidate.summary.mean_cost,
                    threshold=thresholds.max_cost,
                    reason=f"mean_cost {candidate.summary.mean_cost} vs max {thresholds.max_cost}",
                )
            )

        if thresholds.max_latency_ms is None:
            checks.append(GateCheck(name="latency", passed=True, skipped=True, reason="no latency budget"))
        elif candidate.summary.mean_latency_ms is None:
            checks.append(
                GateCheck(
                    name="latency",
                    passed=True,
                    skipped=True,
                    reason="latency metric not present",
                )
            )
        else:
            checks.append(
                GateCheck(
                    name="latency",
                    passed=candidate.summary.mean_latency_ms <= thresholds.max_latency_ms,
                    observed=candidate.summary.mean_latency_ms,
                    threshold=thresholds.max_latency_ms,
                    reason=(
                        f"mean_latency_ms {candidate.summary.mean_latency_ms} "
                        f"vs max {thresholds.max_latency_ms}"
                    ),
                )
            )

        failed = [c for c in checks if not c.skipped and not c.passed]
        passed = not failed
        reason = "gate passed" if passed else "; ".join(c.reason for c in failed)
        return GateDecision(
            passed=passed,
            checks=checks,
            reason=reason,
            experiment_run_id=candidate.run_id,
            baseline_run_id=baseline.run_id if baseline is not None else None,
            details={"pass_rate": candidate.summary.pass_rate, "n_cases": candidate.summary.n_cases},
        )
