// Solar Atelier style reminder: every component is a reusable field instrument—clear state, compact metadata, and one useful next action.

import type { ReactNode } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  BatteryCharging,
  Check,
  CircleAlert,
  CircleGauge,
  Fan,
  Flame,
  Lightbulb,
  Refrigerator,
  Snowflake,
  Sparkles,
  Tv,
  WashingMachine,
  Zap,
} from "lucide-react";
import { Link } from "wouter";
import type { Appliance, DashboardSummary, HealthStatus, Severity } from "@/lib/api";

const applianceIcons: Record<string, typeof Zap> = {
  snowflake: Snowflake,
  refrigerator: Refrigerator,
  flame: Flame,
  tv: Tv,
  "washing-machine": WashingMachine,
  lightbulb: Lightbulb,
  fan: Fan,
};

export function ApplianceIcon({ type, size = 20 }: { type: string; size?: number }) {
  const Icon = applianceIcons[type] ?? Zap;
  return <Icon size={size} strokeWidth={1.8} />;
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: string; actions?: ReactNode }) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && <div className="eyebrow-label"><span className="eyebrow-dot" />{eyebrow}</div>}
        <h2 className="page-title">{title}</h2>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
    </div>
  );
}

export function SectionHeading({ eyebrow, title, link, href = "#" }: { eyebrow?: string; title: string; link?: string; href?: string }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <div className="eyebrow-label eyebrow-label--muted">{eyebrow}</div>}
        <h3>{title}</h3>
      </div>
      {link && <Link href={href} className="text-link">{link}<ArrowUpRight size={14} /></Link>}
    </div>
  );
}

export function StatusBadge({ status, kind = "health" }: { status: string; kind?: "health" | "severity" | "state" }) {
  const normalized = status.toLowerCase().replace(/\s/g, "-");
  return <span className={`status-badge status-badge--${normalized} status-badge--${kind}`}><span className="status-badge__dot" />{status}</span>;
}

export function KpiCard({ icon, label, value, suffix, change, tone = "amber", note }: { icon: ReactNode; label: string; value: string; suffix?: string; change?: number; tone?: string; note?: string }) {
  const positive = (change ?? 0) < 0;
  return (
    <div className={`kpi-card kpi-card--${tone}`}>
      <div className="kpi-card__top"><span className="kpi-icon">{icon}</span>{note && <span className="kpi-note">{note}</span>}</div>
      <div className="kpi-value">{value}<small>{suffix}</small></div>
      <div className="kpi-label">{label}</div>
      {change !== undefined && <div className={`kpi-change ${positive ? "kpi-change--positive" : ""}`}>{positive ? <ArrowDownRight size={14} /> : <ArrowUpRight size={14} />}{Math.abs(change)}% vs last period</div>}
    </div>
  );
}

export function SummaryKpis({ summary }: { summary: DashboardSummary }) {
  return (
    <div className="kpi-grid">
      <KpiCard icon={<Zap size={18} />} label="Current power" value={summary.currentPower.toFixed(2)} suffix=" kW" change={summary.change} tone="amber" note="LIVE" />
      <KpiCard icon={<BatteryCharging size={18} />} label="Today's energy" value={summary.todayEnergy.toFixed(1)} suffix=" kWh" change={-4.2} tone="teal" />
      <KpiCard icon={<CircleGauge size={18} />} label="Estimated bill" value="₹142" change={-6.8} tone="moss" />
      <KpiCard icon={<Fan size={18} />} label="Active appliances" value={String(summary.activeAppliances)} suffix=" / 6" tone="ink" note="NOW" />
      <KpiCard icon={<CircleAlert size={18} />} label="Detected anomalies" value={String(summary.detectedAnomalies)} tone="coral" note="REVIEW" />
      <KpiCard icon={<Sparkles size={18} />} label="Potential monthly savings" value="₹520" change={12.4} tone="violet" />
    </div>
  );
}

export function ApplianceCard({ appliance }: { appliance: Appliance }) {
  const tone = appliance.accent;
  return (
    <Link href={`/appliances/${appliance.id}`} className={`appliance-card appliance-card--${tone}`}>
      <div className="appliance-card__header">
        <span className="appliance-icon"><ApplianceIcon type={appliance.icon} /></span>
        <StatusBadge status={appliance.state} kind="state" />
      </div>
      <div className="appliance-card__name">{appliance.name}</div>
      <div className="appliance-card__power"><strong>{appliance.currentPower.toFixed(2)}</strong><span>kW now</span></div>
      <div className="appliance-card__meter"><span style={{ width: `${Math.max(appliance.contributionPercentage, 3)}%` }} /></div>
      <div className="appliance-card__footer"><span>{appliance.contributionPercentage}% of current use</span><StatusBadge status={appliance.healthStatus} /></div>
    </Link>
  );
}

export function MiniStat({ label, value, accent = "" }: { label: string; value: string; accent?: string }) {
  return <div className="mini-stat"><span>{label}</span><strong className={accent}>{value}</strong></div>;
}

export function EmptyState({ title, description, icon = <Sparkles size={24} /> }: { title: string; description: string; icon?: ReactNode }) {
  return <div className="empty-state"><div className="empty-state__icon">{icon}</div><h3>{title}</h3><p>{description}</p></div>;
}

export function CompletionCheck({ completed }: { completed?: boolean }) {
  return <span className={`completion-check ${completed ? "completion-check--done" : ""}`}>{completed ? <Check size={13} /> : null}</span>;
}

export type { HealthStatus, Severity };
