// Solar Atelier style reminder: pages read like chapters in an energy field report—observations first, controls second, explanations always nearby.

import { useRef, useState } from "react";
import { ArrowLeft, ArrowRight, BellRing, Check, CheckCircle2, ChevronRight, Download, FileBarChart2, FileSpreadsheet, Flame, House, Info, LoaderCircle, LockKeyhole, Mail, Moon, MoreHorizontal, Save, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Target, ThermometerSun, Timer, Trash2, Upload, UserRound, Zap } from "lucide-react";
import { Link, useLocation, useRoute } from "wouter";
import { toast } from "sonner";
import AppShell from "@/components/AppShell";
import { ApplianceCard, ApplianceIcon, CompletionCheck, EmptyState, KpiCard, MiniStat, PageHeader, SectionHeading, StatusBadge } from "@/components/energy/Blocks";
import { ConsumptionChart, DailyBarChart, DistributionChart, ForecastChart } from "@/components/energy/Charts";
import { anomalies, appliances, dailyReadings, distribution, forecast, forecastReadings, home, recommendations as seedRecommendations, user, weeklyReadings, type Appliance } from "@/lib/api";
import { trpc } from "@/lib/trpc";

const reportImage = "/manus-storage/smartenergy-report_9161fc2e.jpg";

function PageFrame({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <AppShell><div className={`workspace-page page-enter ${className}`}>{children}</div></AppShell>;
}

function FilterBar({ items, active, onChange }: { items: string[]; active: string; onChange: (value: string) => void }) {
  return <div className="filter-bar" role="tablist">{items.map((item) => <button key={item} className={active === item ? "is-active" : ""} onClick={() => onChange(item)} role="tab" aria-selected={active === item}>{item}</button>)}</div>;
}

export function AppliancesPage() {
  const [filter, setFilter] = useState("All appliances");
  const filtered = filter === "Active now" ? appliances.filter((item) => item.state === "ON") : filter === "Needs review" ? appliances.filter((item) => item.healthStatus !== "Normal") : appliances;
  return <PageFrame>
    <PageHeader eyebrow="Load inventory / 06 connected" title="Appliance watch" description="A live view of the loads NILM has separated from your aggregate meter signal." actions={<button className="secondary-button" onClick={() => toast.success("Appliance data refreshed from mock service.")}><Zap size={15} /> Refresh readings</button>} />
    <div className="mini-stat-grid"><MiniStat label="Total appliances" value="06" /><MiniStat label="Active now" value="04" accent="text-amber" /><MiniStat label="Today's consumption" value="18.6 kWh" /><MiniStat label="Highest consumer" value="AC / 43%" accent="text-coral" /></div>
    <div className="table-toolbar"><FilterBar items={["All appliances", "Active now", "Needs review"]} active={filter} onChange={setFilter} /><span className="toolbar-note"><span className="status-dot status-dot--teal" /> Updated 2 minutes ago</span></div>
    <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Appliance</th><th>Status</th><th>Current power</th><th>Today's energy</th><th>Daily average</th><th>Contribution</th><th>Health</th><th>Last active</th><th /></tr></thead><tbody>{filtered.map((appliance) => <tr key={appliance.id} onClick={() => window.location.href = `/appliances/${appliance.id}`}><td><div className="table-appliance"><span className={`table-appliance__icon table-appliance__icon--${appliance.accent}`}><ApplianceIcon type={appliance.icon} size={18} /></span><div><strong>{appliance.name}</strong><span>{appliance.category}</span></div></div></td><td><StatusBadge status={appliance.state} kind="state" /></td><td><strong>{appliance.currentPower.toFixed(2)} kW</strong></td><td>{appliance.todayEnergy.toFixed(1)} kWh</td><td>{appliance.averageDailyEnergy.toFixed(1)} kWh</td><td><div className="contribution-cell"><span className="contribution-track"><i style={{ width: `${appliance.contributionPercentage}%` }} /></span><b>{appliance.contributionPercentage}%</b></div></td><td><StatusBadge status={appliance.healthStatus} /></td><td>{appliance.lastActive}</td><td><ChevronRight size={16} className="table-arrow" /></td></tr>)}</tbody></table>{filtered.length === 0 && <EmptyState title="No appliances in this view" description="Try a different filter to see more connected loads." />}</div>
    <div className="appliance-card-grid-mobile">{filtered.map((appliance) => <ApplianceCard appliance={appliance} key={appliance.id} />)}</div>
    <footer className="workspace-footer"><span><CheckCircle2 size={14} /> All connected loads are reporting</span><span>Data source: mock NILM service</span></footer>
  </PageFrame>;
}

export function ApplianceDetailPage() {
  const [, params] = useRoute<{ id: string }>("/appliances/:id");
  const appliance = appliances.find((item) => item.id === params?.id) ?? appliances[0];
  return <PageFrame className="detail-page">
    <Link href="/appliances" className="back-link"><ArrowLeft size={15} /> Back to appliances</Link>
    <div className="detail-hero"><div className={`detail-hero__icon detail-hero__icon--${appliance.accent}`}><ApplianceIcon type={appliance.icon} size={31} /></div><div className="detail-hero__copy"><div className="eyebrow-label eyebrow-label--muted">{appliance.category} / appliance profile</div><h2>{appliance.name}</h2><p>Last signal received just now · {home.name}</p></div><div className="detail-hero__status"><StatusBadge status={appliance.state} kind="state" /><StatusBadge status={appliance.healthStatus} /></div></div>
    <div className="detail-stat-grid"><KpiCard icon={<Zap size={17} />} label="Current power" value={appliance.currentPower.toFixed(2)} suffix=" kW" tone="amber" note="LIVE" /><KpiCard icon={<ThermometerSun size={17} />} label="Today's consumption" value={appliance.todayEnergy.toFixed(1)} suffix=" kWh" tone="teal" /><KpiCard icon={<Target size={17} />} label="Weekly consumption" value="47.8" suffix=" kWh" tone="moss" /><KpiCard icon={<ShieldCheck size={17} />} label="Daily average" value={appliance.averageDailyEnergy.toFixed(1)} suffix=" kWh" tone="ink" /></div>
    <section className="detail-grid"><ConsumptionChart compact /><div className="pattern-card"><div className="panel-kicker">Pattern comparison</div><h3>Normal usage <em>vs</em> current</h3><p className="panel-subtitle">A simple read of today's operating rhythm against recent behavior.</p><div className="pattern-compare"><div className="pattern-row"><span>Morning</span><div className="pattern-bars"><i style={{ width: "38%" }} /><b style={{ width: "29%" }} /></div><small>−9%</small></div><div className="pattern-row"><span>Afternoon</span><div className="pattern-bars"><i style={{ width: "56%" }} /><b style={{ width: "61%" }} /></div><small className="text-coral">+7%</small></div><div className="pattern-row"><span>Evening</span><div className="pattern-bars"><i style={{ width: "62%" }} /><b className="pattern-bars__hot" style={{ width: "86%" }} /></div><small className="text-coral">+24%</small></div></div><div className="pattern-legend"><span><i className="pattern-dot pattern-dot--normal" /> Normal pattern</span><span><i className="pattern-dot pattern-dot--current" /> Current pattern</span></div></div></section>
    <section className="detail-grid detail-grid--two"><DailyBarChart /><div className={`anomaly-callout anomaly-callout--${appliance.healthStatus === "Warning" ? "warning" : "clear"}`}><div className="anomaly-callout__head"><span className="anomaly-callout__icon">{appliance.healthStatus === "Warning" ? <Info size={18} /> : <CheckCircle2 size={18} />}</span><div><div className="panel-kicker">Signal interpretation</div><h3>{appliance.healthStatus === "Warning" ? "Potential anomaly detected" : "No anomaly detected"}</h3></div></div><p>{appliance.healthStatus === "Warning" ? "Potential abnormal energy behavior detected. The evening pattern is above the historical range, but electricity data alone cannot confirm a hardware fault." : "The appliance is behaving within its recent energy range. We'll keep watching for meaningful changes."}</p>{appliance.healthStatus === "Warning" && <div className="anomaly-callout__detail"><span>Observed</span><strong>24% above normal in the evening</strong></div>}<button className="secondary-button" onClick={() => toast.info("Monitoring note added to the mock appliance record.")}>{appliance.healthStatus === "Warning" ? "Keep monitoring" : "Add a note"} <ArrowRight size={14} /></button></div></section>
    <footer className="workspace-footer"><span><CheckCircle2 size={14} /> Appliance detail loaded</span><span>Pattern window: last 30 days</span></footer>
  </PageFrame>;
}

export function AnalyticsPage() {
  const [range, setRange] = useState("7 Days");
  const [appliance, setAppliance] = useState("All appliances");
  return <PageFrame>
    <PageHeader eyebrow="Measured across your home" title="Energy analytics" description="Zoom out from the live signal and see the rhythms behind it." actions={<button className="secondary-button" onClick={() => toast.success("Analytics range refreshed.")}><SlidersHorizontal size={15} /> Customize view</button>} />
    <div className="analytics-controls"><FilterBar items={["Today", "7 Days", "30 Days", "Custom range"]} active={range} onChange={setRange} /><label className="select-field"><span>Show</span><select value={appliance} onChange={(event) => setAppliance(event.target.value)}>{["All appliances", ...appliances.map((item) => item.name)].map((item) => <option key={item}>{item}</option>)}</select><ChevronRight size={14} /></label></div>
    <div className="mini-stat-grid mini-stat-grid--analytics"><MiniStat label="Total consumption" value={range === "Today" ? "18.6 kWh" : "129.4 kWh"} /><MiniStat label="Average daily" value="18.5 kWh" accent="text-teal" /><MiniStat label="Peak consumption" value="5.6 kW" accent="text-amber" /><MiniStat label="Estimated monthly" value="620 kWh" /></div>
    <section className="analytics-grid"><ConsumptionChart /><DistributionChart /></section><section className="analytics-grid analytics-grid--bottom"><DailyBarChart /><div className="weekly-compare"><div className="chart-panel__header"><div><div className="panel-kicker">Month-to-date</div><h3>Weekly comparison</h3></div><span className="chart-callout chart-callout--down">−3.2% <ArrowDownRightIcon /></span></div><div className="week-bars">{weeklyReadings.map((item) => <div className="week-bar" key={item.week}><div className="week-bar__value">{item.actual}</div><div className="week-bar__track"><i style={{ height: `${(item.actual / 160) * 100}%` }} /></div><span>{item.week}</span></div>)}</div><div className="week-compare-note"><span><i className="pattern-dot pattern-dot--current" /> This month</span><span><i className="pattern-dot pattern-dot--normal" /> Typical</span><strong>143 kWh <small>current week</small></strong></div></div></section>
  </PageFrame>;
}

function ArrowDownRightIcon() { return <ArrowRight size={14} className="rotate-45" />; }

export function ForecastPage() {
  return <PageFrame>
    <PageHeader eyebrow="Looking ahead / model outlook" title="Energy forecast" description="A transparent view of what the next few days may look like, based on your recent patterns." actions={<FilterBar items={["Next 24 hours", "Next 7 days", "Next 30 days"]} active="Next 7 days" onChange={(value) => toast.info(`Forecast window set to ${value}.`)} />} />
    <div className="forecast-kpi-grid"><KpiCard icon={<Clock3Icon />} label="Predicted tomorrow" value="21.3" suffix=" kWh" tone="amber" note="24 HRS" /><KpiCard icon={<CalendarIcon />} label="Predicted this week" value="143" suffix=" kWh" tone="teal" /><KpiCard icon={<FileBarChart2 size={18} />} label="Predicted monthly" value="620" suffix=" kWh" tone="moss" /><KpiCard icon={<CircleRupeeIcon />} label="Estimated monthly bill" value="₹4,850" tone="ink" /></div>
    <ForecastChart />
    <section className="forecast-explanation"><div className="forecast-explanation__copy"><div className="eyebrow-label eyebrow-label--muted">Model notes</div><h3>Why is consumption expected to increase?</h3><p>The forecast is not a verdict. It is a planning signal based on repeatable patterns in the mock service.</p></div><div className="forecast-reasons"><div className="forecast-reason"><span>01</span><strong>Higher evening usage</strong><p>Your home typically uses more energy after work hours.</p></div><div className="forecast-reason"><span>02</span><strong>Expected AC runtime</strong><p>Cooling demand is trending higher across the last seven days.</p></div><div className="forecast-reason"><span>03</span><strong>Weekend rhythm</strong><p>Home occupancy usually adds a small weekend lift.</p></div></div></section>
    <footer className="workspace-footer"><span><Sparkles size={14} /> Forecast confidence: 84%</span><span>Mock model output · re-evaluates every 24 hours</span></footer>
  </PageFrame>;
}

function Clock3Icon() { return <Timer size={18} />; }
function CalendarIcon() { return <House size={18} />; }
function CircleRupeeIcon() { return <span className="rupee-icon">₹</span>; }

export function AnomaliesPage() {
  const [selected, setSelected] = useState(anomalies[0]);
  const counts = { total: anomalies.length, high: anomalies.filter((item) => item.severity === "High").length, medium: anomalies.filter((item) => item.severity === "Medium").length, resolved: anomalies.filter((item) => item.status === "Resolved").length };
  return <PageFrame>
    <PageHeader eyebrow="Signal review / 03 observations" title="Fault & anomalies" description="Potential unusual energy behavior, explained without overclaiming what the meter can know." actions={<button className="secondary-button" onClick={() => toast.success("Anomaly feed refreshed.")}><Zap size={15} /> Re-scan signals</button>} />
    <div className="mini-stat-grid anomaly-stat-grid"><MiniStat label="Total anomalies" value={String(counts.total).padStart(2, "0")} /><MiniStat label="High severity" value={String(counts.high).padStart(2, "0")} accent="text-coral" /><MiniStat label="Medium severity" value={String(counts.medium).padStart(2, "0")} accent="text-amber" /><MiniStat label="Resolved" value={String(counts.resolved).padStart(2, "0")} accent="text-teal" /></div>
    <div className="anomalies-layout"><div className="data-table-wrap"><table className="data-table data-table--anomalies"><thead><tr><th>Appliance</th><th>Detected at</th><th>Expected</th><th>Observed</th><th>Severity</th><th>Score</th><th>Status</th></tr></thead><tbody>{anomalies.map((anomaly) => <tr key={anomaly.id} className={selected.id === anomaly.id ? "is-selected" : ""} onClick={() => setSelected(anomaly)}><td><strong>{anomaly.appliance}</strong></td><td>{anomaly.detectedAt}</td><td>{anomaly.expectedPower}</td><td>{anomaly.actualPower}</td><td><StatusBadge status={anomaly.severity} kind="severity" /></td><td><div className="score-cell"><span className="score-track"><i style={{ width: `${anomaly.anomalyScore * 100}%` }} /></span><b>{Math.round(anomaly.anomalyScore * 100)}%</b></div></td><td><StatusBadge status={anomaly.status} /></td></tr>)}</tbody></table></div><aside className="anomaly-detail-panel"><div className="anomaly-detail-panel__top"><div className="panel-kicker">Selected observation</div><button className="icon-button icon-button--soft" aria-label="More anomaly actions" onClick={() => toast.info("More anomaly actions are available when the backend is connected.")}><MoreHorizontal size={17} /></button></div><div className={`anomaly-large-icon anomaly-large-icon--${selected.severity.toLowerCase()}`}><Zap size={20} /></div><h3>{selected.appliance} <StatusBadge status={selected.status} /></h3><p className="anomaly-detail-panel__lead">{selected.explanation}</p><div className="anomaly-facts"><div><span>Expected</span><strong>{selected.expectedPower}</strong></div><div><span>Observed</span><strong>{selected.actualPower}</strong></div><div><span>Duration</span><strong>{selected.duration}</strong></div><div><span>Anomaly score</span><strong>{Math.round(selected.anomalyScore * 100)}%</strong></div></div><div className="anomaly-action"><div className="panel-kicker">Recommended action</div><p>{selected.action}</p></div><button className="primary-button primary-button--full" onClick={() => toast.success("Anomaly marked for monitoring in mock data.")}><ShieldCheck size={15} /> Mark as monitoring</button></aside></div>
    <footer className="workspace-footer"><span><Info size={14} /> Electricity data suggests patterns, not definite hardware failure.</span><span>Last scan: 2 minutes ago</span></footer>
  </PageFrame>;
}

export function RecommendationsPage() {
  const [items, setItems] = useState(seedRecommendations);
  const update = (id: string, action: "complete" | "ignore" | "save") => setItems((current) => current.map((item) => item.id === id ? { ...item, completed: action === "complete" } : item));
  const savings = items.filter((item) => !item.completed).reduce((sum, item) => sum + item.moneySaving, 0);
  return <PageFrame>
    <PageHeader eyebrow="Personalized by your patterns" title="Recommendations" description="Small, specific shifts that could make this home a little lighter on the grid." actions={<button className="secondary-button" onClick={() => toast.info("Recommendations are generated from mock patterns for now.")}><Sparkles size={15} /> How this works</button>} />
    <div className="recommendations-summary"><div className="recommendations-summary__main"><span>Potential monthly savings</span><strong>₹{savings}</strong><small>from the shifts still open</small></div><div className="recommendations-summary__metric"><span>Potential energy reduction</span><strong>32 kWh<span>/month</span></strong></div><div className="recommendations-summary__metric"><span>Active suggestions</span><strong>{items.filter((item) => !item.completed).length}<span> of {items.length}</span></strong></div></div>
    <div className="recommendation-list">{items.map((recommendation, index) => <article className={`recommendation-detail-card ${recommendation.completed ? "recommendation-detail-card--completed" : ""}`} key={recommendation.id}><div className="recommendation-detail-card__number">0{index + 1}</div><div className="recommendation-detail-card__content"><div className="recommendation-detail-card__heading"><div><div className="eyebrow-label eyebrow-label--muted">{recommendation.appliance}</div><h3>{recommendation.title}</h3></div><StatusBadge status={recommendation.priority} kind="severity" /></div><p>{recommendation.reason}</p><div className="recommendation-metrics"><div><span>Energy saving</span><strong>{recommendation.energySaving} kWh<small>/month</small></strong></div><div><span>Money saving</span><strong>₹{recommendation.moneySaving}<small>/month</small></strong></div><div><span>Confidence</span><strong>{Math.round(recommendation.confidence * 100)}%</strong></div></div><div className="recommendation-actions"><button className="quiet-button" onClick={() => { update(recommendation.id, "save"); toast.success("Thanks — we’ll use that signal in the mock experience."); }}><Check size={14} /> Helpful</button><button className="quiet-button" onClick={() => toast.info("Not helpful feedback saved to the mock service.")}>Not helpful</button><button className="primary-button" onClick={() => { update(recommendation.id, "complete"); toast.success("Recommendation marked complete."); }}>{recommendation.completed ? "Completed" : "Mark completed"}</button><button className="icon-button icon-button--soft" onClick={() => { update(recommendation.id, "ignore"); toast.info("Recommendation hidden from the active list."); }} aria-label="Ignore recommendation"><MoreHorizontal size={16} /></button></div></div><CompletionCheck completed={recommendation.completed} /></article>)}</div>
    <footer className="workspace-footer"><span><CheckCircle2 size={14} /> Recommendations are mock outputs for this demo</span><span>They will later come from the FastAPI service</span></footer>
  </PageFrame>;
}

const reportCards = [
  { title: "Monthly energy report", detail: "A clear summary of consumption, bill estimate, and changes.", icon: FileBarChart2, tone: "amber", meta: "August 2026" },
  { title: "Weekly home report", detail: "The last seven days of energy rhythm in one view.", icon: House, tone: "teal", meta: "Week 35" },
  { title: "Appliance report", detail: "Load-by-load energy contribution and health signals.", icon: Zap, tone: "moss", meta: "06 loads" },
  { title: "Anomaly report", detail: "Every potential unusual pattern currently on record.", icon: Info, tone: "coral", meta: "03 observations" },
  { title: "Savings report", detail: "Recommendations, confidence, and estimated opportunity.", icon: Sparkles, tone: "violet", meta: "₹520 opportunity" },
];

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StoredReportsPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const filesQuery = trpc.files.list.useQuery();
  const utils = trpc.useUtils();
  const uploadMutation = trpc.files.upload.useMutation({
    onSuccess: async () => {
      await utils.files.list.invalidate();
      toast.success("File uploaded to your private report library.");
      if (inputRef.current) inputRef.current.value = "";
    },
    onError: (error) => toast.error(error.message || "Upload failed. Please try again."),
  });
  const removeMutation = trpc.files.remove.useMutation({
    onSuccess: async () => {
      await utils.files.list.invalidate();
      toast.success("File removed from your report library.");
    },
    onError: (error) => toast.error(error.message || "Could not remove the file."),
  });

  const handleFile = (file?: File) => {
    if (!file) return;
    if (file.size > 12 * 1024 * 1024) {
      toast.error("Please choose a file smaller than 12 MB.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result ?? "");
      const contentBase64 = dataUrl.split(",")[1] ?? "";
      uploadMutation.mutate({ fileName: file.name, mimeType: file.type || "application/octet-stream", sizeBytes: file.size, contentBase64 });
    };
    reader.onerror = () => toast.error("The file could not be read in this browser.");
    reader.readAsDataURL(file);
  };

  return <section className="storage-panel"><div className="storage-panel__heading"><div><div className="panel-kicker">Private report library</div><h3>Bring your files into the energy story.</h3><p>Upload a PDF, CSV, image, or exported reading. Files are stored privately for your signed-in profile; the database keeps metadata while storage holds the bytes.</p></div><span className="storage-security"><LockKeyhole size={14} /> User-scoped</span></div><input ref={inputRef} className="visually-hidden-input" type="file" accept=".pdf,.csv,.xlsx,.png,.jpg,.jpeg,.webp,.json,text/csv,application/pdf,image/*" onChange={(event) => handleFile(event.target.files?.[0])} /><button className={`storage-dropzone ${uploadMutation.isPending ? "storage-dropzone--pending" : ""}`} onClick={() => inputRef.current?.click()} disabled={uploadMutation.isPending} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleFile(event.dataTransfer.files?.[0]); }}><span className="storage-dropzone__icon">{uploadMutation.isPending ? <LoaderCircle size={21} className="spin" /> : <Upload size={21} />}</span><strong>{uploadMutation.isPending ? "Uploading securely…" : "Choose a file or drop it here"}</strong><span>{uploadMutation.isPending ? "Sending bytes through the server storage layer" : "PDF, CSV, spreadsheet, image, or JSON · up to 12 MB"}</span></button><div className="stored-files-list"><div className="stored-files-list__header"><span>Stored files</span><span>{filesQuery.isLoading ? "Loading…" : `${filesQuery.data?.length ?? 0} files`}</span></div>{filesQuery.isError ? <div className="storage-state storage-state--error"><Info size={16} /><span>We could not load your files. Sign in and try again.</span><button className="text-button" onClick={() => filesQuery.refetch()}>Retry</button></div> : filesQuery.isLoading ? <div className="storage-state"><LoaderCircle size={16} className="spin" /> Loading your private files…</div> : filesQuery.data?.length ? filesQuery.data.map((file) => <div className="stored-file-row" key={file.id}><div className="stored-file-row__icon"><FileBarChart2 size={17} /></div><div className="stored-file-row__copy"><strong>{file.fileName}</strong><span>{file.mimeType} · {formatBytes(file.sizeBytes)} · {new Date(file.createdAt).toLocaleDateString()}</span></div><a className="icon-button icon-button--soft" href={file.url} target="_blank" rel="noreferrer" aria-label={`Open ${file.fileName}`}><ArrowRight size={15} /></a><button className="icon-button icon-button--soft" disabled={removeMutation.isPending} onClick={() => removeMutation.mutate({ id: file.id })} aria-label={`Remove ${file.fileName}`}><Trash2 size={15} /></button></div>) : <div className="storage-state"><FileBarChart2 size={17} /><span>No files yet. Your uploaded reports will appear here.</span></div>}</div></section>;
}

export function ReportsPage() {
  return <PageFrame>
    <PageHeader eyebrow="Exportable snapshots" title="Reports" description="Presentation-ready summaries of how this home is using energy." actions={<button className="secondary-button" onClick={() => toast.success("Mock CSV export prepared.")}><Download size={15} /> Export all</button>} />
    <div className="reports-feature"><div className="reports-feature__image" style={{ backgroundImage: `url(${reportImage})` }} /><div className="reports-feature__copy"><div className="eyebrow-label"><span className="eyebrow-dot" /> Featured snapshot</div><h3>August energy report</h3><p>A visual summary of this month's usage, appliance contributions, anomaly signals, and opportunities to save.</p><div className="reports-feature__meta"><span>Last generated 30 Aug 2026</span><span>12 pages</span></div><div className="button-row"><button className="primary-button" onClick={() => toast.info("Report preview is a mock interaction for now.")}>View report <ArrowRight size={15} /></button><button className="quiet-button" onClick={() => toast.success("Mock PDF download started.")}><Download size={15} /> Download PDF</button></div></div></div>
    <div className="report-grid">{reportCards.map(({ title, detail, icon: Icon, tone, meta }) => <article className="report-card" key={title}><div className={`report-card__icon report-card__icon--${tone}`}><Icon size={19} /></div><div className="report-card__meta">{meta}</div><h3>{title}</h3><p>{detail}</p><div className="report-card__actions"><button className="text-button" onClick={() => toast.info(`${title} preview is a mock interaction.`)}>View report <ArrowRight size={14} /></button><button className="icon-button icon-button--soft" aria-label={`Export ${title}`} onClick={() => toast.success("Mock CSV export prepared.")}><FileSpreadsheet size={15} /></button></div></article>)}</div>
    <StoredReportsPanel />
  </PageFrame>;
}

export function SettingsPage() {
  const [tariff, setTariff] = useState("8.50");
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(true);
  return <PageFrame>
    <PageHeader eyebrow="Workspace preferences" title="Settings" description="Keep the intelligence layer aligned with the way your home is measured." actions={<button className="primary-button" onClick={() => toast.success("Settings saved to the mock profile.")}><Save size={15} /> Save changes</button>} />
    <div className="settings-layout"><aside className="settings-nav"><span className="settings-nav__label">Preferences</span>{[{ label: "Profile settings", icon: UserRound }, { label: "Home settings", icon: House }, { label: "Appliance settings", icon: Settings2 }, { label: "Notifications", icon: BellRing }, { label: "Energy tariff", icon: Zap }, { label: "Theme settings", icon: Moon }].map(({ label, icon: Icon }, index) => <button className={`settings-nav__item ${index === 0 ? "is-active" : ""}`} key={label}><Icon size={16} />{label}<ChevronRight size={14} /></button>)}</aside><div className="settings-content"><section className="settings-card"><div className="settings-card__heading"><div><div className="panel-kicker">Profile settings</div><h3>Your profile</h3></div><span className="settings-card__tag">Local mock profile</span></div><div className="profile-form"><div className="profile-form__avatar avatar avatar--large">{user.initials}</div><label><span>Full name</span><input defaultValue="Abhishek S." /></label><label><span>Email</span><input defaultValue={user.email} type="email" /></label></div></section><section className="settings-card"><div className="settings-card__heading"><div><div className="panel-kicker">Home settings</div><h3>Make the model more relevant</h3></div></div><div className="form-grid"><label><span>Home name</span><input defaultValue={home.name} /></label><label><span>City</span><input defaultValue={home.city} /></label><label><span>Electricity rate</span><div className="input-with-suffix"><span>₹</span><input value={tariff} onChange={(event) => setTariff(event.target.value)} /><small>/ kWh</small></div></label><label><span>Currency</span><select defaultValue="INR"><option value="INR">Indian Rupee (₹)</option><option value="USD">US Dollar ($)</option></select></label></div></section><section className="settings-card"><div className="settings-card__heading"><div><div className="panel-kicker">Notification settings</div><h3>Choose what deserves a ping</h3></div></div><div className="toggle-list"><label className="toggle-row"><div><strong>Anomaly alerts</strong><span>Notify when a potential unusual pattern is detected.</span></div><input type="checkbox" checked={notifications} onChange={() => setNotifications((value) => !value)} /><i /></label><label className="toggle-row"><div><strong>Weekly energy note</strong><span>A short summary every Monday morning.</span></div><input type="checkbox" defaultChecked /><i /></label><label className="toggle-row"><div><strong>Recommendation updates</strong><span>Let the model surface new energy-saving shifts.</span></div><input type="checkbox" defaultChecked /><i /></label></div></section><section className="settings-card settings-card--last"><div className="settings-card__heading"><div><div className="panel-kicker">Theme settings</div><h3>Set the mood</h3></div></div><label className="toggle-row"><div><strong>Dark interface</strong><span>Use a lower-light canvas for evening review sessions.</span></div><input type="checkbox" checked={darkMode} onChange={() => { setDarkMode((value) => !value); toast.info("Theme preference is recorded in the mock profile."); }} /><i /></label></section></div></div>
  </PageFrame>;
}

function AuthFrame({ children, mode }: { children: React.ReactNode; mode: "login" | "register" }) {
  return <div className="auth-page"><div className="auth-visual" style={{ backgroundImage: "url('/manus-storage/smartenergy-hero_7f081891.jpg')" }}><div className="auth-visual__veil" /><div className="auth-visual__copy"><Link href="/" className="auth-brand"><span className="brand-mark"><span className="brand-mark__arc" /><span className="brand-mark__trace brand-mark__trace--one" /><span className="brand-mark__trace brand-mark__trace--two" /><span className="brand-mark__trace brand-mark__trace--three" /></span><strong>SmartEnergy</strong></Link><div className="auth-quote"><span className="eyebrow-label"><span className="eyebrow-dot" /> Home intelligence</span><h1>See what your home is telling you.</h1><p>Transform aggregate energy readings into a clearer, more useful picture of everyday life.</p></div><div className="auth-proof"><span><CheckCircle2 size={15} /> Mock FastAPI-ready architecture</span><span><CheckCircle2 size={15} /> Privacy-first household signals</span></div></div></div><main className="auth-form-area"><Link href="/" className="auth-back"><ArrowLeft size={15} /> Back to overview</Link><div className="auth-form-wrap">{children}<div className="auth-switch">{mode === "login" ? <>New to SmartEnergy? <Link href="/register">Create an account</Link></> : <>Already have an account? <Link href="/login">Sign in</Link></>}</div><p className="auth-disclaimer">Demo mode · authentication is mocked and ready for a future JWT connection.</p></div></main></div>;
}

export function LoginPage() {
  return <AuthFrame mode="login"><div className="auth-eyebrow">Your home / return visit</div><h2>Welcome back.</h2><p className="auth-intro">Sign in to pick up your home's latest energy story.</p><form className="auth-form" onSubmit={(event) => { event.preventDefault(); toast.success("Mock sign-in successful."); window.location.href = "/"; }}><label><span>Email address</span><div className="auth-input"><Mail size={17} /><input type="email" defaultValue="abhishek@example.com" /></div></label><label><span>Password</span><div className="auth-input"><LockKeyhole size={17} /><input type="password" defaultValue="password" /></div></label><div className="auth-form__row"><label className="checkbox-label"><input type="checkbox" defaultChecked /> Remember me</label><button type="button" className="text-button" onClick={() => toast.info("Password reset will connect to the auth service later.")}>Forgot password?</button></div><button type="submit" className="primary-button primary-button--full">Sign in <ArrowRight size={15} /></button></form></AuthFrame>;
}

export function RegisterPage() {
  return <AuthFrame mode="register"><div className="auth-eyebrow">Start with your home</div><h2>Create your profile.</h2><p className="auth-intro">Set up a personal view of the energy behind your everyday routines.</p><form className="auth-form" onSubmit={(event) => { event.preventDefault(); toast.success("Mock profile created."); window.location.href = "/"; }}><label><span>Name</span><div className="auth-input"><UserRound size={17} /><input defaultValue="Abhishek S." /></div></label><label><span>Email address</span><div className="auth-input"><Mail size={17} /><input type="email" defaultValue="abhishek@example.com" /></div></label><label><span>Password</span><div className="auth-input"><LockKeyhole size={17} /><input type="password" defaultValue="password" /></div></label><label><span>Confirm password</span><div className="auth-input"><LockKeyhole size={17} /><input type="password" defaultValue="password" /></div></label><button type="submit" className="primary-button primary-button--full">Create profile <ArrowRight size={15} /></button></form></AuthFrame>;
}
