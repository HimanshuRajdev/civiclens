import { Component, OnInit, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { Stats } from '../../models/complaint.model';
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">Analytics & Insights</h1>
          <p class="page-subtitle">Deep-dive into complaint trends, department performance, and city health metrics</p>
        </div>
      </div>

      <!-- Top Metrics Row -->
      <div class="metrics-row">
        <div class="metric-card" *ngFor="let m of metrics; let i = index" [style.animation-delay]="i * 0.08 + 's'">
          <div class="metric-ring" [style.background]="m.ringBg">
            <span class="metric-icon" [innerHTML]="m.icon"></span>
          </div>
          <div class="metric-info">
            <span class="metric-value">{{ m.value }}</span>
            <span class="metric-label">{{ m.label }}</span>
          </div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="charts-grid">
        <div class="card chart-large">
          <div class="card-header">
            <h3>Complaints Over Time</h3>
            <div class="time-filters">
              <button class="time-btn" [class.active]="timeRange === '7d'" (click)="setTimeRange('7d')">7D</button>
              <button class="time-btn" [class.active]="timeRange === '30d'" (click)="setTimeRange('30d')">30D</button>
              <button class="time-btn" [class.active]="timeRange === '90d'" (click)="setTimeRange('90d')">90D</button>
            </div>
          </div>
          <div class="chart-container-lg">
            <canvas #timelineChart></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3>Department Workload</h3>
          </div>
          <div class="chart-container-md">
            <canvas #deptChart></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3>Severity Trends</h3>
          </div>
          <div class="chart-container-md">
            <canvas #severityTrendChart></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3>Issue Type Radar</h3>
          </div>
          <div class="chart-container-md">
            <canvas #radarChart></canvas>
          </div>
        </div>
      </div>

      <!-- Department Performance Table -->
      <div class="card perf-card">
        <div class="card-header">
          <h3>Department Performance Scorecard</h3>
        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Department</th>
              <th>Total Cases</th>
              <th>Open</th>
              <th>In Progress</th>
              <th>Resolved</th>
              <th>Resolution Rate</th>
              <th>Performance</th>
            </tr>
          </thead>
          <tbody>
            <tr *ngFor="let d of deptPerformance">
              <td><span class="dept-name-cell">{{ d.department }}</span></td>
              <td>{{ d.total }}</td>
              <td><span class="badge badge-open">{{ d.open }}</span></td>
              <td><span class="badge badge-in-progress">{{ d.inProgress }}</span></td>
              <td><span class="badge badge-resolved">{{ d.resolved }}</span></td>
              <td>
                <div class="rate-bar">
                  <div class="rate-fill" [style.width.%]="d.rate" [style.background]="d.rate > 60 ? 'var(--success)' : d.rate > 30 ? 'var(--warning)' : 'var(--danger)'"></div>
                </div>
                <span class="rate-text">{{ d.rate }}%</span>
              </td>
              <td>
                <span class="perf-badge" [ngClass]="d.rate > 60 ? 'perf-good' : d.rate > 30 ? 'perf-avg' : 'perf-poor'">
                  {{ d.rate > 60 ? 'Good' : d.rate > 30 ? 'Average' : 'Needs Attention' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
  styles: [`
    .page-header { margin-bottom: 24px; }

    .metrics-row {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      margin-bottom: 24px;
    }

    .metric-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      animation: slideUp 0.4s ease forwards;
      opacity: 0;
      transition: all 0.2s ease;
      &:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
    }

    .metric-ring {
      width: 42px;
      height: 42px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .metric-info {
      display: flex;
      flex-direction: column;
    }

    .metric-value {
      font-size: 22px;
      font-weight: 800;
      color: var(--text-primary);
      letter-spacing: -0.3px;
    }

    .metric-label {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
    }

    .charts-grid {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }

    .chart-large { grid-column: 1 / -1; }

    .chart-container-lg { height: 280px; position: relative; }
    .chart-container-md { height: 240px; position: relative; }

    .time-filters {
      display: flex;
      gap: 4px;
      background: var(--bg-surface-2);
      border-radius: var(--radius-sm);
      padding: 2px;
    }

    .time-btn {
      padding: 5px 12px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 600;
      border-radius: 4px;
      cursor: pointer;
      font-family: var(--font-family);
      transition: all 0.15s ease;

      &.active {
        background: var(--accent-blue);
        color: white;
      }
      &:hover:not(.active) {
        color: var(--text-primary);
      }
    }

    .perf-card { padding: 0; overflow: hidden; }
    .perf-card .card-header { padding: 20px 24px; }

    .dept-name-cell {
      font-weight: 600;
      color: var(--text-primary);
    }

    .rate-bar {
      width: 80px;
      height: 5px;
      background: var(--bg-surface-2);
      border-radius: 3px;
      overflow: hidden;
      display: inline-block;
      margin-right: 8px;
      vertical-align: middle;
    }

    .rate-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 1s ease;
    }

    .rate-text {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .perf-badge {
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;

      &.perf-good { background: var(--success-dim); color: var(--success); }
      &.perf-avg { background: var(--warning-dim); color: var(--warning); }
      &.perf-poor { background: var(--danger-dim); color: var(--danger); }
    }

    @media (max-width: 1200px) {
      .metrics-row { grid-template-columns: repeat(3, 1fr); }
      .charts-grid { grid-template-columns: 1fr; }
    }
  `]
})
export class AnalyticsComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('timelineChart') timelineChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('deptChart') deptChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('severityTrendChart') severityTrendRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('radarChart') radarChartRef!: ElementRef<HTMLCanvasElement>;

  stats: Stats | null = null;
  timeRange = '30d';
  metrics: any[] = [];
  deptPerformance: any[] = [];
  private charts: Chart[] = [];

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadData();
  }

  ngAfterViewInit() {
    setTimeout(() => this.buildCharts(), 500);
  }

  ngOnDestroy() {
    this.charts.forEach(c => c.destroy());
  }

  setTimeRange(range: string) {
    this.timeRange = range;
    this.buildCharts();
  }

  loadData() {
    this.api.getStats().subscribe({
      next: (s) => {
        this.stats = s;
        this.buildMetrics(s);
        this.buildDeptPerformance(s);
        this.buildCharts();
      },
      error: () => this.useMockData(),
    });
  }

  private buildMetrics(s: Stats) {
    const resolved = s.by_status['Resolved'] || 0;
    const rate = s.total > 0 ? Math.round((resolved / s.total) * 100) : 0;
    this.metrics = [
      { label: 'Total Reports', value: s.total, icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>', ringBg: 'linear-gradient(135deg, #3b82f6, #2563eb)' },
      { label: 'High Severity', value: s.by_severity['High'] || 0, icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>', ringBg: 'linear-gradient(135deg, #ef4444, #dc2626)' },
      { label: 'Resolution Rate', value: rate + '%', icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>', ringBg: 'linear-gradient(135deg, #10b981, #059669)' },
      { label: 'Avg Response', value: '2.4h', icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>', ringBg: 'linear-gradient(135deg, #f59e0b, #d97706)' },
      { label: 'Departments', value: '5', icon: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg>', ringBg: 'linear-gradient(135deg, #8b5cf6, #7c3aed)' },
    ];
  }

  private buildDeptPerformance(s: Stats) {
    const deptMap: Record<string, string> = {
      'Pothole': 'Roads & Infrastructure', 'Sinkhole': 'Roads & Infrastructure',
      'Water Leakage': 'Water & Sewage', 'Garbage Overflow': 'Sanitation',
      'Broken Streetlight': 'Electrical', 'Broken Sidewalk': 'Public Works',
    };

    // Simplified: distribute complaints proportionally
    const depts: Record<string, { total: number; open: number; inProgress: number; resolved: number }> = {};
    for (const [type, count] of Object.entries(s.by_type)) {
      const dept = deptMap[type] || type;
      if (!depts[dept]) depts[dept] = { total: 0, open: 0, inProgress: 0, resolved: 0 };
      depts[dept].total += count;
      // Estimate distribution
      const ratio = count / s.total;
      depts[dept].open += Math.round((s.by_status['Open'] || 0) * ratio);
      depts[dept].inProgress += Math.round((s.by_status['In Progress'] || 0) * ratio);
      depts[dept].resolved += Math.round((s.by_status['Resolved'] || 0) * ratio);
    }

    this.deptPerformance = Object.entries(depts).map(([department, d]) => ({
      department,
      total: d.total,
      open: d.open,
      inProgress: d.inProgress,
      resolved: d.resolved,
      rate: d.total > 0 ? Math.round((d.resolved / d.total) * 100) : 0,
    }));
  }

  private buildCharts() {
    if (!this.stats) return;
    this.charts.forEach(c => c.destroy());
    this.charts = [];
    const s = this.stats;

    // Timeline (area chart)
    if (this.timelineChartRef?.nativeElement) {
      const days = this.timeRange === '7d' ? 7 : this.timeRange === '30d' ? 30 : 90;
      const labels = Array.from({ length: days }, (_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - (days - 1 - i));
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      });
      const baseRate = s.total / days;
      const data = labels.map(() => Math.max(0, Math.round(baseRate + (Math.random() - 0.4) * baseRate * 2)));

      this.charts.push(new Chart(this.timelineChartRef.nativeElement, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Complaints',
            data,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: '#3b82f6',
            borderWidth: 2,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' }, maxTicksLimit: 10 },
              border: { display: false },
            },
            y: {
              grid: { color: 'rgba(42,42,62,0.5)' },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } },
              border: { display: false },
              beginAtZero: true,
            }
          },
          interaction: { mode: 'index', intersect: false },
        }
      }));
    }

    // Department workload horizontal bar
    if (this.deptChartRef?.nativeElement) {
      const deptLabels = this.deptPerformance.map(d => d.department);
      const deptData = this.deptPerformance.map(d => d.total);
      this.charts.push(new Chart(this.deptChartRef.nativeElement, {
        type: 'bar',
        data: {
          labels: deptLabels,
          datasets: [{
            data: deptData,
            backgroundColor: ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444'],
            borderRadius: 6,
            borderSkipped: false,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { color: 'rgba(42,42,62,0.5)' },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } },
              border: { display: false },
            },
            y: {
              grid: { display: false },
              ticks: { color: '#8585a0', font: { size: 10, family: 'Inter' } },
              border: { display: false },
            }
          }
        }
      }));
    }

    // Severity trend (stacked bar)
    if (this.severityTrendRef?.nativeElement) {
      const weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
      const high = s.by_severity['High'] || 0;
      const med = s.by_severity['Medium'] || 0;
      const low = s.by_severity['Low'] || 0;

      this.charts.push(new Chart(this.severityTrendRef.nativeElement, {
        type: 'bar',
        data: {
          labels: weeks,
          datasets: [
            { label: 'High', data: [Math.round(high * 0.3), Math.round(high * 0.25), Math.round(high * 0.2), Math.round(high * 0.25)], backgroundColor: '#ef4444', borderRadius: 4 },
            { label: 'Medium', data: [Math.round(med * 0.2), Math.round(med * 0.3), Math.round(med * 0.25), Math.round(med * 0.25)], backgroundColor: '#f59e0b', borderRadius: 4 },
            { label: 'Low', data: [Math.round(low * 0.25), Math.round(low * 0.25), Math.round(low * 0.3), Math.round(low * 0.2)], backgroundColor: '#3b82f6', borderRadius: 4 },
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { color: '#8585a0', font: { size: 10, family: 'Inter' }, usePointStyle: true, pointStyleWidth: 8, padding: 12 } }
          },
          scales: {
            x: { stacked: true, grid: { display: false }, ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } }, border: { display: false } },
            y: { stacked: true, grid: { color: 'rgba(42,42,62,0.5)' }, ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } }, border: { display: false } }
          }
        }
      }));
    }

    // Radar chart
    if (this.radarChartRef?.nativeElement) {
      const types = Object.keys(s.by_type).map(t => t.replace('_', ' '));
      const values = Object.values(s.by_type);
      this.charts.push(new Chart(this.radarChartRef.nativeElement, {
        type: 'radar',
        data: {
          labels: types,
          datasets: [{
            label: 'Issue Count',
            data: values,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.15)',
            borderWidth: 2,
            pointBackgroundColor: '#8b5cf6',
            pointRadius: 4,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            r: {
              grid: { color: 'rgba(42,42,62,0.5)' },
              angleLines: { color: 'rgba(42,42,62,0.5)' },
              pointLabels: { color: '#8585a0', font: { size: 10, family: 'Inter' } },
              ticks: { display: false },
            }
          }
        }
      }));
    }
  }

  private useMockData() {
    this.stats = {
      total: 47,
      by_type: { 'Pothole': 15, 'Water Leakage': 10, 'Garbage Overflow': 8, 'Broken Streetlight': 6, 'Sinkhole': 5, 'Broken Sidewalk': 3 },
      by_status: { 'Open': 18, 'In Progress': 14, 'Resolved': 15 },
      by_severity: { 'High': 20, 'Medium': 14, 'Low': 13 },
    };
    this.buildMetrics(this.stats);
    this.buildDeptPerformance(this.stats);
    setTimeout(() => this.buildCharts(), 300);
  }
}
