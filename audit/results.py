"""Structured audit findings (errors vs warnings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Severity = Literal["error", "warning", "info"]


@dataclass
class AuditFinding:
    severity: Severity
    code: str
    message: str


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str) -> None:
        self.findings.append(AuditFinding(severity=severity, code=code, message=message))

    def extend(self, other: AuditReport) -> None:
        """Append another report's findings (mutates self)."""
        self.findings.extend(other.findings)

    def errors(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "error"]

    def warnings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    def raise_for_errors(self) -> None:
        errs = self.errors()
        if errs:
            msg = "; ".join(f"{e.code}: {e.message}" for e in errs)
            raise ValueError(f"Audit failed: {msg}")


def merge_audit_reports(*reports: AuditReport) -> AuditReport:
    """Combine multiple reports into one (shallow copy of findings)."""
    out = AuditReport()
    for r in reports:
        out.extend(r)
    return out


def audit_findings_as_jsonable(rep: AuditReport) -> list[dict[str, Any]]:
    """Serialize findings for ``audit.json`` / experiment bundles."""
    return [{"severity": f.severity, "code": f.code, "message": f.message} for f in rep.findings]
