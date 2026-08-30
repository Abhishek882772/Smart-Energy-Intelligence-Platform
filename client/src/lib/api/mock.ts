// Mock data and API interface for SmartEnergy frontend

export type HealthStatus = "Normal" | "Warning" | "Critical";
export type Severity = "Low" | "Medium" | "High" | "Critical";
export type ApplianceState = "ON" | "OFF" | "STANDBY";

export interface DashboardSummary {
  currentPower: number;
  change: number;
  todayEnergy: number;
  activeAppliances: number;
  detectedAnomalies: number;
}

export interface Appliance {
  id: string;
  name: string;
  category: string;
  icon: string;
  accent: string;
  state: string;
  currentPower: number;
  todayEnergy: number;
  averageDailyEnergy: number;
  contributionPercentage: number;
  healthStatus: HealthStatus | string;
  lastActive: string;
}

export interface Anomaly {
  id: string;
  appliance: string;
  detectedAt: string;
  expectedPower: string;
  actualPower: string;
  severity: Severity | string;
  anomalyScore: number;
  status: string;
  duration: string;
  explanation: string;
  action: string;
}

export interface Recommendation {
  id: string;
  title: string;
  appliance: string;
  reason: string;
  priority: string;
  energySaving: number;
  moneySaving: number;
  confidence: number;
  completed?: boolean;
}

export interface MeterReading {
  time: string;
  actual: number;
  predicted: number;
}

export interface DailyReading {
  day: string;
  consumption: number;
  average: number;
}

export interface WeeklyReading {
  week: string;
  actual: number;
  average: number;
}

export interface DistributionItem {
  name: string;
  value: number;
  color: string;
}

export interface ForecastReading {
  day: string;
  actual?: number;
  forecast?: number;
  low?: number;
  high?: number;
}

export const user = {
  name: "Abhishek",
  email: "abhishek@example.com",
  initials: "AS",
};

export const home = {
  name: "Greenfield Haven",
  city: "Bengaluru, India",
};

export const forecast = {
  summary: "Expected weekly consumption 143 kWh",
  confidence: 0.84,
};

export const dashboardSummary: DashboardSummary = {
  currentPower: 4.25,
  change: -8.4,
  todayEnergy: 18.6,
  activeAppliances: 4,
  detectedAnomalies: 2,
};

export const appliances: Appliance[] = [
  {
    id: "living-room-ac",
    name: "Living Room AC",
    category: "Cooling",
    icon: "snowflake",
    accent: "amber",
    state: "ON",
    currentPower: 1.85,
    todayEnergy: 8.0,
    averageDailyEnergy: 7.2,
    contributionPercentage: 43,
    healthStatus: "Warning",
    lastActive: "Active now",
  },
  {
    id: "water-heater",
    name: "Water Heater (Geyser)",
    category: "Heating",
    icon: "flame",
    accent: "teal",
    state: "ON",
    currentPower: 1.2,
    todayEnergy: 4.5,
    averageDailyEnergy: 4.1,
    contributionPercentage: 24,
    healthStatus: "Normal",
    lastActive: "Active now",
  },
  {
    id: "refrigerator",
    name: "Kitchen Refrigerator",
    category: "Refrigeration",
    icon: "refrigerator",
    accent: "moss",
    state: "ON",
    currentPower: 0.35,
    todayEnergy: 2.1,
    averageDailyEnergy: 2.2,
    contributionPercentage: 11,
    healthStatus: "Normal",
    lastActive: "Active now",
  },
  {
    id: "induction-cooktop",
    name: "Induction Cooktop",
    category: "Cooking",
    icon: "flame",
    accent: "coral",
    state: "OFF",
    currentPower: 0.0,
    todayEnergy: 1.8,
    averageDailyEnergy: 1.9,
    contributionPercentage: 10,
    healthStatus: "Normal",
    lastActive: "45m ago",
  },
  {
    id: "washing-machine",
    name: "Washing Machine",
    category: "Laundry",
    icon: "washing-machine",
    accent: "ink",
    state: "STANDBY",
    currentPower: 0.05,
    todayEnergy: 1.2,
    averageDailyEnergy: 1.4,
    contributionPercentage: 7,
    healthStatus: "Normal",
    lastActive: "2h ago",
  },
  {
    id: "entertainment-tv",
    name: "TV & Entertainment",
    category: "Entertainment",
    icon: "tv",
    accent: "violet",
    state: "ON",
    currentPower: 0.25,
    todayEnergy: 1.0,
    averageDailyEnergy: 1.1,
    contributionPercentage: 5,
    healthStatus: "Normal",
    lastActive: "Active now",
  },
];

export const anomalies: Anomaly[] = [
  {
    id: "anom-1",
    appliance: "Living Room AC",
    detectedAt: "Today, 18:45",
    expectedPower: "1.40 kW",
    actualPower: "2.15 kW",
    severity: "High",
    anomalyScore: 0.88,
    status: "Active",
    duration: "45 mins",
    explanation: "Evening compressor cycling rate is 24% higher than expected baseline.",
    action: "Inspect air filter and verify thermostat setpoint during peak evening hours.",
  },
  {
    id: "anom-2",
    appliance: "Water Heater (Geyser)",
    detectedAt: "Yesterday, 21:10",
    expectedPower: "0.00 kW",
    actualPower: "1.50 kW",
    severity: "Medium",
    anomalyScore: 0.74,
    status: "Monitoring",
    duration: "1 hr 20m",
    explanation: "Late-night heating cycle detected outside regular morning schedule.",
    action: "Check automated timer schedule to avoid unnecessary overnight standby heating.",
  },
  {
    id: "anom-3",
    appliance: "Kitchen Refrigerator",
    detectedAt: "28 Aug, 14:20",
    expectedPower: "0.25 kW",
    actualPower: "0.55 kW",
    severity: "Low",
    anomalyScore: 0.61,
    status: "Resolved",
    duration: "30 mins",
    explanation: "Slightly prolonged defrost cycle following frequent door openings.",
    action: "Clean condenser coils if duty cycle remains elevated over 48 hours.",
  },
];

export const recommendations: Recommendation[] = [
  {
    id: "rec-1",
    title: "Shift geyser heating to off-peak morning hours",
    appliance: "Water Heater",
    reason: "Water heating between 7–9 PM coincides with higher peak demand and higher grid strain.",
    priority: "High",
    energySaving: 18,
    moneySaving: 240,
    confidence: 0.91,
    completed: false,
  },
  {
    id: "rec-2",
    title: "Adjust AC thermostat by +1°C (24°C to 25°C)",
    appliance: "Living Room AC",
    reason: "A single degree shift reduces compressor continuous load by up to 6% without sacrificing comfort.",
    priority: "Medium",
    energySaving: 12,
    moneySaving: 160,
    confidence: 0.86,
    completed: false,
  },
  {
    id: "rec-3",
    title: "Run laundry loads on eco-cycle during afternoon solar window",
    appliance: "Washing Machine",
    reason: "Running full cold-water wash loads between 12–3 PM optimizes overall household peak balance.",
    priority: "Low",
    energySaving: 8,
    moneySaving: 120,
    confidence: 0.79,
    completed: false,
  },
];

export const meterReadings: MeterReading[] = [
  { time: "00:00", actual: 1.2, predicted: 1.4 },
  { time: "03:00", actual: 0.9, predicted: 1.1 },
  { time: "06:00", actual: 2.1, predicted: 1.8 },
  { time: "09:00", actual: 3.4, predicted: 3.1 },
  { time: "12:00", actual: 3.8, predicted: 3.6 },
  { time: "15:00", actual: 3.2, predicted: 3.5 },
  { time: "18:00", actual: 4.8, predicted: 4.2 },
  { time: "21:00", actual: 4.25, predicted: 3.9 },
];

export const dailyReadings: DailyReading[] = [
  { day: "Mon", consumption: 16.4, average: 17.8 },
  { day: "Tue", consumption: 19.1, average: 18.2 },
  { day: "Wed", consumption: 17.5, average: 18.0 },
  { day: "Thu", consumption: 20.2, average: 18.5 },
  { day: "Fri", consumption: 18.6, average: 18.1 },
  { day: "Sat", consumption: 21.4, average: 19.5 },
  { day: "Sun", consumption: 22.1, average: 20.0 },
];

export const weeklyReadings: WeeklyReading[] = [
  { week: "W31", actual: 124, average: 130 },
  { week: "W32", actual: 138, average: 132 },
  { week: "W33", actual: 146, average: 135 },
  { week: "W34", actual: 141, average: 134 },
  { week: "W35", actual: 143, average: 138 },
];

export const distribution: DistributionItem[] = [
  { name: "AC & Cooling", value: 43, color: "#F5A524" },
  { name: "Water Heating", value: 24, color: "#2B9DA5" },
  { name: "Kitchen & Cooking", value: 16, color: "#E05D44" },
  { name: "Refrigeration", value: 10, color: "#5B8C5A" },
  { name: "Lighting & Media", value: 7, color: "#7E6B8F" },
];

export const forecastReadings: ForecastReading[] = [
  { day: "Aug 26", actual: 16.4, low: 14.5, high: 18.5 },
  { day: "Aug 27", actual: 19.1, low: 16.0, high: 21.0 },
  { day: "Aug 28", actual: 17.5, low: 15.0, high: 19.5 },
  { day: "Aug 29", actual: 20.2, low: 17.0, high: 22.0 },
  { day: "Aug 30", actual: 18.6, forecast: 18.6, low: 16.5, high: 21.5 },
  { day: "Aug 31", forecast: 21.3, low: 18.0, high: 24.5 },
  { day: "Sep 01", forecast: 20.8, low: 17.5, high: 23.8 },
  { day: "Sep 02", forecast: 19.5, low: 16.0, high: 22.5 },
  { day: "Sep 03", forecast: 20.1, low: 17.0, high: 23.0 },
  { day: "Sep 04", forecast: 22.4, low: 19.0, high: 25.5 },
  { day: "Sep 05", forecast: 21.9, low: 18.5, high: 25.0 },
];

export async function getDashboardSummary(): Promise<DashboardSummary> {
  return dashboardSummary;
}

export async function getAppliances(): Promise<Appliance[]> {
  return appliances;
}

export async function getRecommendations(): Promise<Recommendation[]> {
  return recommendations;
}
