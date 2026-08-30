// Solar Atelier style reminder: the route map is one continuous analytical workspace, never a collection of disconnected screens.

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import { AnalyticsPage, AnomaliesPage, ApplianceDetailPage, AppliancesPage, ForecastPage, LoginPage, RecommendationsPage, RegisterPage, ReportsPage, SettingsPage } from "./pages/WorkspacePages";
import NotFound from "./pages/NotFound";
function Router() {
  // make sure to consider if you need authentication for certain routes
  return <Switch>
    <Route path="/" component={Home} />
    <Route path="/appliances/:id" component={ApplianceDetailPage} />
    <Route path="/appliances" component={AppliancesPage} />
    <Route path="/analytics" component={AnalyticsPage} />
    <Route path="/forecast" component={ForecastPage} />
    <Route path="/anomalies/:id" component={AnomaliesPage} />
    <Route path="/anomalies" component={AnomaliesPage} />
    <Route path="/recommendations" component={RecommendationsPage} />
    <Route path="/reports" component={ReportsPage} />
    <Route path="/settings" component={SettingsPage} />
    <Route path="/login" component={LoginPage} />
    <Route path="/register" component={RegisterPage} />
    <Route path="/404" component={NotFound} />
    <Route component={NotFound} />
  </Switch>;
}

export default function App() {
  return <ErrorBoundary><ThemeProvider defaultTheme="light"><TooltipProvider><Toaster position="bottom-right" /><Router /></TooltipProvider></ThemeProvider></ErrorBoundary>;
}
