// Solar Atelier style reminder: treat the shell as a field notebook spine—deep ink rail, warm workspace, signal-led navigation.

import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Activity,
  BarChart3,
  Bell,
  CalendarDays,
  ChartNoAxesCombined,
  FileText,
  LayoutDashboard,
  Lightbulb,
  LogOut,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings,
  Sparkles,
  TriangleAlert,
  X,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { user } from "@/lib/api/mock";

const navItems = [
  { label: "Dashboard", href: "/", icon: LayoutDashboard },
  { label: "Appliances", href: "/appliances", icon: Zap },
  { label: "Energy analytics", href: "/analytics", icon: ChartNoAxesCombined },
  { label: "Forecast", href: "/forecast", icon: CalendarDays },
  { label: "Fault & anomalies", href: "/anomalies", icon: TriangleAlert },
  { label: "Recommendations", href: "/recommendations", icon: Lightbulb },
  { label: "Reports", href: "/reports", icon: FileText },
  { label: "Settings", href: "/settings", icon: Settings },
];

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/appliances": "Appliances",
  "/analytics": "Energy analytics",
  "/forecast": "Forecast",
  "/anomalies": "Fault & anomalies",
  "/recommendations": "Recommendations",
  "/reports": "Reports",
  "/settings": "Settings",
};

function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand-lockup">
      <div className="brand-mark" aria-hidden="true">
        <span className="brand-mark__arc" />
        <span className="brand-mark__trace brand-mark__trace--one" />
        <span className="brand-mark__trace brand-mark__trace--two" />
        <span className="brand-mark__trace brand-mark__trace--three" />
      </div>
      {!compact && (
        <div className="brand-copy">
          <span className="brand-name">SmartEnergy</span>
          <span className="brand-subtitle">Home intelligence</span>
        </div>
      )}
    </div>
  );
}

function Sidebar({ collapsed, onCollapse }: { collapsed: boolean; onCollapse: () => void }) {
  const [location] = useLocation();
  return (
    <aside className={`app-sidebar ${collapsed ? "app-sidebar--collapsed" : ""}`}>
      <div className="sidebar-topline" />
      <div className="sidebar-header">
        <Link href="/" aria-label="SmartEnergy dashboard">
          <BrandMark compact={collapsed} />
        </Link>
        <button className="icon-button icon-button--ghost sidebar-collapse" onClick={onCollapse} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      <div className="sidebar-section-label">Workspace</div>
      <nav className="sidebar-nav" aria-label="Primary navigation">
        {navItems.map(({ label, href, icon: Icon }) => {
          const active = href === "/" ? location === "/" : location.startsWith(href);
          return (
            <Link key={href} href={href} className={`sidebar-link ${active ? "sidebar-link--active" : ""}`}>
              <span className="sidebar-link__icon"><Icon size={18} strokeWidth={active ? 2.3 : 1.8} /></span>
              <span className="sidebar-link__label">{label}</span>
              {label === "Fault & anomalies" && <span className="sidebar-link__count">2</span>}
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-flow">
        <div className="sidebar-flow__eyebrow"><Activity size={13} /> Intelligence flow</div>
        <div className="flow-steps">
          <div className="flow-step flow-step--done"><span>01</span> Meter data</div>
          <div className="flow-connector" />
          <div className="flow-step flow-step--done"><span>02</span> NILM layer</div>
          <div className="flow-connector" />
          <div className="flow-step flow-step--current"><span>03</span> Live insights</div>
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="sidebar-profile">
          <div className="avatar avatar--amber">{user.initials}</div>
          <div className="sidebar-profile__copy">
            <strong>{user.name} S.</strong>
            <span>Home owner</span>
          </div>
          <button className="icon-button icon-button--ghost" aria-label="Open profile menu" onClick={() => toast.info("Profile menu is ready for the auth connection.")}><Settings size={16} /></button>
        </div>
        <button className="sidebar-logout" onClick={() => toast.info("Mock logout — authentication can connect here later.")}><LogOut size={16} /> <span>Sign out</span></button>
      </div>
    </aside>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const isDetail = location.startsWith("/appliances/");
  const title = isDetail ? "Appliance detail" : pageTitles[location] ?? "SmartEnergy";

  return (
    <div className="app-shell">
      <div className={`mobile-sidebar-backdrop ${mobileOpen ? "mobile-sidebar-backdrop--visible" : ""}`} onClick={() => setMobileOpen(false)} />
      <div className={`sidebar-wrap ${mobileOpen ? "sidebar-wrap--open" : ""}`}>
        <Sidebar collapsed={collapsed} onCollapse={() => setCollapsed((value) => !value)} />
        <button className="mobile-sidebar-close icon-button" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={19} /></button>
      </div>
      <main className={`app-main ${collapsed ? "app-main--sidebar-collapsed" : ""}`}>
        <header className="top-header">
          <div className="top-header__leading">
            <button className="mobile-menu-button icon-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
            <div>
              <div className="breadcrumb"><span>SmartEnergy</span><span className="breadcrumb__slash">/</span><strong>{title}</strong></div>
              <h1 className="top-header__title">{title}</h1>
            </div>
          </div>
          <div className="top-header__actions">
            <label className="global-search">
              <Search size={17} />
              <input aria-label="Search SmartEnergy" placeholder="Search insights" />
              <kbd>⌘ K</kbd>
            </label>
            <button className="icon-button notification-button" aria-label="Notifications" onClick={() => toast.info("You have 2 anomaly alerts to review.")}>
              <Bell size={18} />
              <span className="notification-dot" />
            </button>
            <div className="header-user"><div className="avatar avatar--small">{user.initials}</div><span>{user.name}</span></div>
          </div>
        </header>
        <div className="workspace-scroll">{children}</div>
      </main>
    </div>
  );
}

export { BrandMark, navItems };
