// Solar Atelier style reminder: lead with a calm editorial narrative, then let the charts and alerts explain what the home is telling us.

import { useEffect, useState } from "react";
import { ArrowRight, CheckCircle2, Clock3, Lightbulb, RefreshCw, ShieldCheck, Sparkles, TriangleAlert, Wind, Zap } from "lucide-react";
import { Link } from "wouter";
import { toast } from "sonner";
import AppShell from "@/components/AppShell";
import { ApplianceCard, EmptyState, SectionHeading, StatusBadge, SummaryKpis } from "@/components/energy/Blocks";
import { ConsumptionChart, DistributionChart } from "@/components/energy/Charts";
import { anomalies, getAppliances, getDashboardSummary, getRecommendations, home, type Appliance, type DashboardSummary, type Recommendation } from "@/lib/api";

const heroImage = "/manus-storage/smartenergy-hero_7f081891.jpg";

export default function Home() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [applianceList, setApplianceList] = useState<Appliance[]>([]);
  const [recommendationList, setRecommendationList] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;
    Promise.all([getDashboardSummary(), getAppliances(), getRecommendations()]).then(([summaryData, applianceData, recommendationData]) => {
      if (!mounted) return;
      setSummary(summaryData);
      setApplianceList(applianceData);
      setRecommendationList(recommendationData);
      window.setTimeout(() => mounted && setLoading(false), 220);
    });
    return () => { mounted = false; };
  }, []);

  return <AppShell>
    <div className="dashboard-page page-enter">
      <section className="dashboard-welcome">
        <div className="welcome-copy">
          <div className="eyebrow-label"><span className="eyebrow-dot eyebrow-dot--live" />Friday, 30 August 2026 <span className="eyebrow-divider" /> Bengaluru, India</div>
          <h2>Good morning, <em>Abhishek.</em></h2>
          <p>Here's your home's energy overview. One clear signal at a time.</p>
          <div className="welcome-actions"><Link href="/analytics" className="primary-button">Explore analytics <ArrowRight size={16} /></Link><button className="quiet-button" onClick={() => toast.success("Dashboard refreshed from the mock service.")}><RefreshCw size={15} /> Refresh data</button></div>
        </div>
        <div className="welcome-visual" style={{ backgroundImage: `url(${heroImage})` }}>
          <div className="welcome-visual__overlay" />
          <div className="welcome-visual__label"><span className="signal-pulse" /> Home signal healthy</div>
          <div className="welcome-visual__caption">A quieter view of a smarter home.</div>
        </div>
      </section>

      <section className="signal-ribbon" aria-label="Energy intelligence workflow">
        <div className="signal-ribbon__label"><Zap size={14} /> Pipeline</div>
        <div className="signal-ribbon__item"><span>01</span><strong>Smart meter</strong><i>Connected</i></div><div className="signal-ribbon__line" />
        <div className="signal-ribbon__item"><span>02</span><strong>NILM layer</strong><i>Decomposed</i></div><div className="signal-ribbon__line" />
        <div className="signal-ribbon__item"><span>03</span><strong>Appliance data</strong><i>Live</i></div><div className="signal-ribbon__line" />
        <div className="signal-ribbon__item signal-ribbon__item--accent"><span>04</span><strong>Actionable insight</strong><i>Ready</i></div>
      </section>

      {loading || !summary ? <div className="loading-surface"><div className="loading-line loading-line--wide" /><div className="loading-line" /><div className="loading-grid">{Array.from({ length: 6 }).map((_, index) => <div className="skeleton-card" key={index} />)}</div></div> : <>
        <section className="dashboard-section"><SectionHeading eyebrow="At a glance" title="The numbers that matter now" link="View full analytics" href="/analytics" /><SummaryKpis summary={summary} /></section>

        <section className="dashboard-section dashboard-grid-main">
          <ConsumptionChart />
          <aside className="insight-stack">
            <div className="insight-card insight-card--amber"><div className="insight-card__top"><span className="insight-icon"><Sparkles size={17} /></span><span className="field-stamp">AI NOTE  /  01</span></div><h3>Evening load is your clearest opportunity.</h3><p>Between 6–9 PM, cooling and water heating account for <strong>71%</strong> of live demand.</p><Link href="/recommendations" className="insight-link">See the suggested shifts <ArrowRight size={14} /></Link></div>
            <div className="insight-card insight-card--quiet"><div className="insight-card__top"><span className="insight-icon"><ShieldCheck size={17} /></span><span className="field-stamp">SYSTEM STATUS</span></div><div className="system-status-row"><div><strong>Model confidence</strong><span>Last updated 2 min ago</span></div><div className="confidence-ring"><b>84</b><small>%</small></div></div><div className="confidence-bar"><span style={{ width: "84%" }} /></div><p>Readings are behaving within the expected household pattern.</p></div>
          </aside>
        </section>

        <section className="dashboard-section"><SectionHeading eyebrow="Appliance watch" title="What is running at home" link="Manage appliances" href="/appliances" /><div className="appliance-grid">{applianceList.slice(0, 5).map((appliance) => <ApplianceCard appliance={appliance} key={appliance.id} />)}<Link href="/appliances" className="appliance-more"><span className="appliance-more__icon"><ArrowRight size={18} /></span><strong>See all appliances</strong><span>6 connected loads</span></Link></div></section>

        <section className="dashboard-section dashboard-grid-secondary"><DistributionChart /><div className="alerts-panel"><SectionHeading eyebrow="Needs a look" title="Recent anomalies" link="Review all" href="/anomalies" /><div className="alert-list">{anomalies.slice(0, 2).map((anomaly) => <div className="alert-item" key={anomaly.id}><div className={`alert-severity alert-severity--${anomaly.severity.toLowerCase()}`}><TriangleAlert size={16} /></div><div className="alert-item__copy"><div className="alert-item__title"><strong>{anomaly.appliance}</strong><StatusBadge status={anomaly.severity} kind="severity" /></div><p>{anomaly.explanation}</p><div className="alert-item__meta"><Clock3 size={13} /> {anomaly.detectedAt} <span /> Score {Math.round(anomaly.anomalyScore * 100)}%</div></div><Link href={`/anomalies/${anomaly.id}`} className="icon-button icon-button--soft" aria-label={`View ${anomaly.appliance} alert`}><ArrowRight size={15} /></Link></div>)}</div></div></section>

        <section className="dashboard-section recommendations-section"><div className="recommendations-intro"><div className="section-heading section-heading--stacked"><div><div className="eyebrow-label eyebrow-label--muted">Personalized by your patterns</div><h3>Small changes, meaningful savings.</h3><p>These recommendations are based on your home's recent appliance-level behavior.</p></div></div><div className="savings-callout"><span>Potential monthly savings</span><strong>₹520</strong><small>if all three shifts stick</small></div></div><div className="recommendation-grid">{recommendationList.map((recommendation) => <div className="recommendation-card" key={recommendation.id}><div className="recommendation-card__top"><span className="recommendation-icon"><Lightbulb size={17} /></span><StatusBadge status={recommendation.priority} kind="severity" /></div><h4>{recommendation.title}</h4><div className="recommendation-card__appliance"><Wind size={13} /> {recommendation.appliance}</div><p>{recommendation.reason}</p><div className="recommendation-card__saving"><span>Potential saving</span><strong>₹{recommendation.moneySaving}<small>/month</small></strong></div><button className="text-button" onClick={() => toast.success("Recommendation saved to your mock action list.")}>Save this shift <ArrowRight size={14} /></button></div>)}</div><Link href="/recommendations" className="recommendations-footer">View all recommendations <ArrowRight size={15} /></Link></section>
      </>}

      <footer className="workspace-footer"><span><CheckCircle2 size={14} /> Mock data service connected</span><span>Built for a clearer energy future · {home.name}</span></footer>
    </div>
  </AppShell>;
}
