import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent),
  },
  {
    path: 'complaints',
    loadComponent: () =>
      import('./pages/complaints/complaints.component').then(m => m.ComplaintsComponent),
  },
  {
    path: 'analytics',
    loadComponent: () =>
      import('./pages/analytics/analytics.component').then(m => m.AnalyticsComponent),
  },
  {
    path: 'map',
    loadComponent: () =>
      import('./pages/map/map.component').then(m => m.MapPageComponent),
  },
  {
    path: 'ai-insights',
    loadComponent: () =>
      import('./pages/ai-insights/ai-insights.component').then(m => m.AiInsightsComponent),
  },
  { path: '**', redirectTo: 'dashboard' },
];
