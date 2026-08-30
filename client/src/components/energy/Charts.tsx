// Solar Atelier style reminder: charts are quiet instruments—warm canvas, ink axes, amber actuals, teal forecasts, and labels that explain the signal.

import { useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ArrowDownRight, ArrowUpRight, Info } from "lucide-react";
import { dailyReadings, distribution, forecastReadings, meterReadings, weeklyReadings } from "@/lib/api";

const chartTooltipStyle = { backgroundColor: "#202526", border: "0", borderRadius: "10px", color: "#fff", fontSize: "12px", padding: "10px 12px" };

function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return <div className="chart-legend">{items.map((item) => <span key={item.label}><i style={{ background: item.color, borderTop: item.dashed ? `2px dashed ${item.color}` : undefined }} />{item.label}</span>)}</div>;
}

export function ConsumptionChart({ compact = false }: { compact?: boolean }) {
  const [range, setRange] = useState("Hourly");
  const data = range === "Daily" ? dailyReadings.map((item) => ({ time: item.day, actual: item.consumption / 4.4, predicted: item.average / 4.4 })) : range === "Weekly" ? weeklyReadings.map((item) => ({ time: item.week, actual: item.actual / 30, predicted: item.average / 30 })) : meterReadings;
  return (
    <div className={`chart-panel ${compact ? "chart-panel--compact" : ""}`}>
      <div className="chart-panel__header">
        <div><div className="panel-kicker">Consumption pattern</div><h3>Power over time</h3><p className="panel-subtitle">Aggregate meter readings, decomposed through NILM.</p></div>
        <div className="segmented-control" role="tablist" aria-label="Consumption range">{["Hourly", "Daily", "Weekly", "Monthly"].map((item) => <button key={item} className={range === item ? "is-active" : ""} onClick={() => setRange(item)} role="tab" aria-selected={range === item}>{item}</button>)}</div>
      </div>
      <div className="chart-meta-row"><span><strong>{range === "Hourly" ? "4.25 kW" : range === "Daily" ? "18.6 kWh" : "143 kWh"}</strong> current view</span><span className="chart-meta-positive"><ArrowDownRight size={14} /> 8.4% vs average</span></div>
      <div className="chart-wrap"><ResponsiveContainer width="100%" height={compact ? 230 : 300}>
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
          <defs><linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#F5A524" stopOpacity={0.24} /><stop offset="100%" stopColor="#F5A524" stopOpacity={0.01} /></linearGradient></defs>
          <CartesianGrid vertical={false} stroke="#E6E0D5" strokeDasharray="4 5" />
          <XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} dy={8} interval="preserveStartEnd" />
          <YAxis tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} tickFormatter={(value) => `${value}kW`} width={42} />
          <Tooltip contentStyle={chartTooltipStyle} formatter={(value, name) => [`${Number(value).toFixed(2)} kW`, name === "actual" ? "Actual" : "Predicted"]} labelStyle={{ color: "#B9C1B8", marginBottom: 4 }} />
          <Area type="monotone" dataKey="actual" stroke="#F5A524" strokeWidth={2.5} fill="url(#actualFill)" activeDot={{ r: 4, fill: "#F5A524", stroke: "#202526", strokeWidth: 2 }} />
          <Line type="monotone" dataKey="predicted" stroke="#2B9DA5" strokeWidth={2} strokeDasharray="5 5" dot={false} />
        </AreaChart>
      </ResponsiveContainer></div>
      <ChartLegend items={[{ label: "Actual consumption", color: "#F5A524" }, { label: "Predicted baseline", color: "#2B9DA5", dashed: true }]} />
    </div>
  );
}

export function DistributionChart() {
  return (
    <div className="chart-panel distribution-panel">
      <div className="chart-panel__header"><div><div className="panel-kicker">Appliance contribution</div><h3>Where energy goes</h3></div><button className="icon-button icon-button--soft" aria-label="About energy distribution"><Info size={16} /></button></div>
      <div className="donut-wrap"><ResponsiveContainer width="100%" height={220}><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={68} outerRadius={93} paddingAngle={3} stroke="none">{distribution.map((entry) => <Cell key={entry.name} fill={entry.color} />)}</Pie><Tooltip contentStyle={chartTooltipStyle} formatter={(value) => [`${value}%`, "Share"]} /></PieChart></ResponsiveContainer><div className="donut-center"><strong>18.6</strong><span>kWh today</span></div></div>
      <div className="distribution-list">{distribution.map((item) => <div className="distribution-row" key={item.name}><span><i style={{ background: item.color }} />{item.name}</span><strong>{item.value}%</strong></div>)}</div>
    </div>
  );
}

export function ForecastChart({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`chart-panel ${compact ? "chart-panel--compact" : ""}`}>
      <div className="chart-panel__header"><div><div className="panel-kicker">Model outlook</div><h3>Historical vs forecast</h3><p className="panel-subtitle">The shaded band shows the expected confidence range.</p></div><span className="confidence-pill">84% confidence</span></div>
      <div className="chart-wrap"><ResponsiveContainer width="100%" height={compact ? 220 : 320}><ComposedChart data={forecastReadings} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
        <defs><linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2B9DA5" stopOpacity={0.18} /><stop offset="100%" stopColor="#2B9DA5" stopOpacity={0.03} /></linearGradient></defs>
        <CartesianGrid vertical={false} stroke="#E6E0D5" strokeDasharray="4 5" />
        <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} dy={8} interval="preserveStartEnd" />
        <YAxis tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} tickFormatter={(value) => `${value}`} width={32} />
        <Tooltip contentStyle={chartTooltipStyle} formatter={(value, name) => [value ? `${Number(value).toFixed(1)} kWh` : "—", name === "actual" ? "Actual" : name === "forecast" ? "Forecast" : name]} />
        <Area type="monotone" dataKey="high" stroke="none" fill="url(#confidenceFill)" activeDot={false} />
        <Area type="monotone" dataKey="low" stroke="none" fill="#F4F0E8" activeDot={false} />
        <Line type="monotone" dataKey="actual" stroke="#202526" strokeWidth={2.5} connectNulls={false} dot={{ r: 2.5, fill: "#202526", strokeWidth: 0 }} />
        <Line type="monotone" dataKey="forecast" stroke="#2B9DA5" strokeWidth={2.5} strokeDasharray="6 5" connectNulls={false} dot={{ r: 3, fill: "#2B9DA5", stroke: "#F8F5EF", strokeWidth: 2 }} />
        <ReferenceLine x="Aug 30" stroke="#F5A524" strokeDasharray="3 4" label={{ value: "TODAY", fill: "#B67813", fontSize: 10, position: "insideTopRight" }} />
      </ComposedChart></ResponsiveContainer></div>
      <ChartLegend items={[{ label: "Historical", color: "#202526" }, { label: "Forecast", color: "#2B9DA5", dashed: true }, { label: "Confidence range", color: "#A6D3D3" }]} />
    </div>
  );
}

export function DailyBarChart() {
  return <div className="chart-panel chart-panel--compact"><div className="chart-panel__header"><div><div className="panel-kicker">Seven day view</div><h3>Daily consumption</h3></div><span className="chart-callout"><ArrowUpRight size={14} /> 4.8%</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height={230}><BarChart data={dailyReadings} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}><CartesianGrid vertical={false} stroke="#E6E0D5" strokeDasharray="4 5" /><XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} dy={8} /><YAxis tickLine={false} axisLine={false} tick={{ fill: "#8D918C", fontSize: 11 }} width={34} /><Tooltip contentStyle={chartTooltipStyle} formatter={(value, name) => [`${Number(value).toFixed(1)} kWh`, name === "consumption" ? "Consumption" : "Average"]} /><Bar dataKey="consumption" fill="#F5A524" radius={[4, 4, 1, 1]} barSize={18} /><Bar dataKey="average" fill="#D9E8E5" radius={[4, 4, 1, 1]} barSize={18} /></BarChart></ResponsiveContainer></div><ChartLegend items={[{ label: "Your home", color: "#F5A524" }, { label: "Typical day", color: "#D9E8E5" }]} /></div>;
}

export function MiniSparkline({ values, color = "#F5A524" }: { values: number[]; color?: string }) {
  const data = values.map((value, index) => ({ index, value }));
  return <div className="mini-sparkline"><ResponsiveContainer width="100%" height="100%"><LineChart data={data}><Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div>;
}
