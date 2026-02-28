import { Component, OnInit, OnDestroy, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../services/api.service';
import { Complaint, Stats } from '../../models/complaint.model';
import { environment } from '../../environments/environment';
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">Command Center</h1>
          <p class="page-subtitle">Real-time civic infrastructure monitoring powered by AI</p>
        </div>
        <div class="header-actions">
          <div class="live-indicator">
            <span class="live-dot"></span>
            <span>Live</span>
          </div>
          <button class="btn btn-secondary" (click)="refreshData()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh
          </button>
        </div>
      </div>

      <!-- KPI Cards -->
      <div class="kpi-grid">
        <div class="kpi-card" *ngFor="let kpi of kpiCards; let i = index" [style.animation-delay]="i * 0.08 + 's'">
          <div class="kpi-icon" [style.background]="kpi.bgColor">
            <span [innerHTML]="kpi.icon"></span>
          </div>
          <div class="kpi-content">
            <span class="kpi-label">{{ kpi.label }}</span>
            <span class="kpi-value">{{ kpi.value }}</span>
          </div>
          <div class="kpi-trend" [class.trend-up]="kpi.trend > 0" [class.trend-down]="kpi.trend < 0">
            <span *ngIf="kpi.trend !== 0">{{ kpi.trend > 0 ? '+' : '' }}{{ kpi.trend }}%</span>
          </div>
        </div>
      </div>

      <!-- Charts Row -->
      <div class="charts-row">
        <div class="card chart-card">
          <div class="card-header">
            <h3>Issue Distribution</h3>
            <span class="card-badge">By Type</span>
          </div>
          <div class="chart-container">
            <canvas #issueChart></canvas>
          </div>
        </div>

        <div class="card chart-card">
          <div class="card-header">
            <h3>Severity Breakdown</h3>
            <span class="card-badge">Overview</span>
          </div>
          <div class="chart-container">
            <canvas #severityChart></canvas>
          </div>
        </div>

        <div class="card chart-card">
          <div class="card-header">
            <h3>Status Pipeline</h3>
            <span class="card-badge">Progress</span>
          </div>
          <div class="chart-container">
            <canvas #statusChart></canvas>
          </div>
        </div>
      </div>

      <!-- Recent Complaints + Quick Stats -->
      <div class="bottom-row">
        <div class="card recent-card">
          <div class="card-header">
            <h3>Recent Complaints</h3>
            <a routerLink="/complaints" class="view-all-link">View All &rarr;</a>
          </div>
          <div class="recent-list" *ngIf="recentComplaints.length > 0">
            <div class="recent-item" *ngFor="let c of recentComplaints; let i = index" [style.animation-delay]="i * 0.05 + 's'" (click)="selectComplaint(c)" style="cursor:pointer">
              <div class="recent-icon" [ngClass]="'severity-' + c.severity.toLowerCase()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              </div>
              <div class="recent-info">
                <span class="recent-type">{{ c.issue_type }}</span>
                <span class="recent-desc">{{ c.description | slice:0:60 }}{{ c.description.length > 60 ? '...' : '' }}</span>
              </div>
              <div class="recent-meta">
                <span class="badge" [ngClass]="{
                  'badge-open': c.status === 'Open',
                  'badge-in-progress': c.status === 'In Progress',
                  'badge-resolved': c.status === 'Resolved'
                }">{{ c.status }}</span>
                <span class="recent-time">{{ getTimeAgo(c.created_at) }}</span>
              </div>
            </div>
          </div>
          <div class="empty-state" *ngIf="recentComplaints.length === 0 && !loading">
            <div class="empty-icon">📋</div>
            <h4>No complaints yet</h4>
            <p>Complaints will appear here once reported via the mobile app</p>
          </div>
        </div>

        <div class="quick-stats-col">
          <div class="card stat-card">
            <div class="stat-header">
              <h3>Department Load</h3>
            </div>
            <div class="dept-bars">
              <div class="dept-bar" *ngFor="let dept of departmentLoad">
                <div class="dept-info">
                  <span class="dept-name">{{ dept.name }}</span>
                  <span class="dept-count">{{ dept.count }}</span>
                </div>
                <div class="dept-progress">
                  <div class="dept-fill" [style.width.%]="dept.percent" [style.background]="dept.color"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="card stat-card resolution-card">
            <div class="stat-header">
              <h3>Resolution Rate</h3>
            </div>
            <div class="resolution-ring">
              <svg viewBox="0 0 120 120" class="ring-svg">
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--border-color)" stroke-width="8"/>
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--success)" stroke-width="8"
                  [attr.stroke-dasharray]="resolutionDash"
                  stroke-dashoffset="0"
                  stroke-linecap="round"
                  transform="rotate(-90 60 60)"/>
              </svg>
              <div class="ring-value">{{ resolutionRate }}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Complaint Detail Modal -->
    <div class="modal-overlay" *ngIf="selectedComplaint" (click)="closeModal()">
      <div class="modal-panel" (click)="$event.stopPropagation()">
        <div class="modal-header">
          <div class="modal-title-row">
            <span class="modal-issue-type">{{ selectedComplaint.issue_type }}</span>
            <span class="badge" [ngClass]="{
              'badge-open': selectedComplaint.status === 'Open',
              'badge-in-progress': selectedComplaint.status === 'In Progress',
              'badge-resolved': selectedComplaint.status === 'Resolved'
            }">{{ selectedComplaint.status }}</span>
          </div>
          <span class="modal-id">{{ selectedComplaint.id }}</span>
          <button class="modal-close" (click)="closeModal()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
        </div>
        <div class="modal-body">
          <div class="modal-image-wrap" *ngIf="selectedComplaint.image_url">
            <img [src]="getImageUrl(selectedComplaint.image_url)" alt="Complaint photo" class="modal-image" />
          </div>
          <div class="modal-fields">
            <div class="modal-field full-width">
              <label>Description</label>
              <p>{{ selectedComplaint.description }}</p>
            </div>
            <div class="modal-field">
              <label>Severity</label>
              <span class="badge" [ngClass]="'badge-' + selectedComplaint.severity.toLowerCase()">{{ selectedComplaint.severity }}</span>
            </div>
            <div class="modal-field">
              <label>Department</label>
              <p>{{ selectedComplaint.department }}</p>
            </div>
            <div class="modal-field">
              <label>Location</label>
              <p>{{ selectedComplaint.lat.toFixed(4) }}, {{ selectedComplaint.lng.toFixed(4) }}</p>
            </div>
            <div class="modal-field">
              <label>Filed At</label>
              <p>{{ selectedComplaint.created_at }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      margin-bottom: 28px;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .live-indicator {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: var(--radius-md);
      color: var(--success);
      font-size: 12px;
      font-weight: 600;
    }

    .live-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.5s ease-in-out infinite;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }

    .kpi-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      padding: 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      transition: all 0.2s ease;
      animation: slideUp 0.4s ease forwards;
      opacity: 0;

      &:hover {
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
      }
    }

    .kpi-icon {
      width: 44px;
      height: 44px;
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .kpi-content {
      display: flex;
      flex-direction: column;
      flex: 1;
    }

    .kpi-label {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .kpi-value {
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: var(--text-primary);
    }

    .kpi-trend {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;

      &.trend-up {
        color: var(--success);
        background: var(--success-dim);
      }
      &.trend-down {
        color: var(--danger);
        background: var(--danger-dim);
      }
    }

    /* Charts Row */
    .charts-row {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }

    .chart-card {
      min-height: 280px;
    }

    .chart-container {
      position: relative;
      height: 220px;
    }

    .card-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-surface-2);
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    /* Bottom Row */
    .bottom-row {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 16px;
    }

    .recent-card {
      max-height: 400px;
      display: flex;
      flex-direction: column;
    }

    .recent-list {
      overflow-y: auto;
      flex: 1;
    }

    .recent-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px 0;
      border-bottom: 1px solid var(--border-light);
      animation: slideUp 0.3s ease forwards;
      opacity: 0;

      &:last-child { border-bottom: none; }
    }

    .recent-icon {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;

      &.severity-high { background: var(--danger-dim); color: var(--danger); }
      &.severity-medium { background: var(--warning-dim); color: var(--warning); }
      &.severity-low { background: var(--accent-blue-dim); color: var(--accent-blue); }
    }

    .recent-info {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }

    .recent-type {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .recent-desc {
      font-size: 12px;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .recent-meta {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 4px;
    }

    .recent-time {
      font-size: 11px;
      color: var(--text-muted);
    }

    .view-all-link {
      font-size: 12px;
      color: var(--accent-blue);
      text-decoration: none;
      font-weight: 500;
      &:hover { text-decoration: underline; }
    }

    /* Quick Stats */
    .quick-stats-col {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .stat-card {
      flex: 1;
    }

    .stat-header h3 {
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 16px;
    }

    .dept-bars {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .dept-bar .dept-info {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .dept-name {
      font-size: 12px;
      color: var(--text-secondary);
    }

    .dept-count {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .dept-progress {
      height: 6px;
      background: var(--bg-surface-2);
      border-radius: 3px;
      overflow: hidden;
    }

    .dept-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 1s ease;
    }

    /* Resolution Ring */
    .resolution-card {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .resolution-ring {
      position: relative;
      width: 120px;
      height: 120px;
    }

    .ring-svg {
      width: 100%;
      height: 100%;
    }

    .ring-value {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 24px;
      font-weight: 800;
      color: var(--success);
    }

    @media (max-width: 1200px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .charts-row { grid-template-columns: 1fr; }
      .bottom-row { grid-template-columns: 1fr; }
    }

    /* Modal */
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(4px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      animation: fadeIn 0.15s ease;
    }

    .modal-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-lg);
      width: 560px;
      max-width: calc(100vw - 40px);
      max-height: calc(100vh - 60px);
      overflow-y: auto;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.5);
      animation: slideUp 0.2s ease;
    }

    .modal-header {
      padding: 20px 20px 16px;
      border-bottom: 1px solid var(--border-color);
      position: relative;
    }

    .modal-title-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 4px;
    }

    .modal-issue-type {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .modal-id {
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 12px;
      color: var(--accent-blue);
      font-weight: 600;
    }

    .modal-close {
      position: absolute;
      top: 16px;
      right: 16px;
      background: var(--bg-surface-2);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-sm);
      color: var(--text-muted);
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      &:hover { color: var(--text-primary); border-color: var(--text-muted); }
    }

    .modal-body {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .modal-image-wrap {
      width: 100%;
      max-height: 260px;
      border-radius: var(--radius-md);
      overflow: hidden;
    }

    .modal-image {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .modal-fields {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .modal-field {
      &.full-width { grid-column: 1 / -1; }

      label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--text-muted);
        font-weight: 600;
        display: block;
        margin-bottom: 4px;
      }

      p {
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
      }
    }
  `]
})
export class DashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('issueChart') issueChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('severityChart') severityChartRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('statusChart') statusChartRef!: ElementRef<HTMLCanvasElement>;

  stats: Stats | null = null;
  recentComplaints: Complaint[] = [];
  loading = true;
  kpiCards: any[] = [];
  departmentLoad: any[] = [];
  resolutionRate = 0;
  resolutionDash = '0 314';
  selectedComplaint: Complaint | null = null;

  private charts: Chart[] = [];
  private refreshInterval: any;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.refreshData();
    this.refreshInterval = setInterval(() => this.refreshData(), 30000);
  }

  ngAfterViewInit() {
    setTimeout(() => this.buildCharts(), 500);
  }

  ngOnDestroy() {
    if (this.refreshInterval) clearInterval(this.refreshInterval);
    this.charts.forEach(c => c.destroy());
  }

  refreshData() {
    this.api.getStats().subscribe({
      next: (s) => {
        this.stats = s;
        this.buildKpis(s);
        this.buildDeptLoad(s);
        this.calculateResolution(s);
        this.buildCharts();
      },
      error: () => this.useMockData(),
    });

    this.api.getComplaints().subscribe({
      next: (c) => {
        this.recentComplaints = c.slice(0, 8);
        this.loading = false;
      },
      error: () => {
        this.recentComplaints = [];
        this.loading = false;
      },
    });
  }

  private buildKpis(s: Stats) {
    this.kpiCards = [
      {
        label: 'Total Reports',
        value: s.total,
        icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
        bgColor: 'linear-gradient(135deg, #3b82f6, #2563eb)',
        trend: 12,
      },
      {
        label: 'Open Issues',
        value: s.by_status['Open'] || 0,
        icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
        bgColor: 'linear-gradient(135deg, #ef4444, #dc2626)',
        trend: -5,
      },
      {
        label: 'In Progress',
        value: s.by_status['In Progress'] || 0,
        icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
        bgColor: 'linear-gradient(135deg, #f59e0b, #d97706)',
        trend: 8,
      },
      {
        label: 'Resolved',
        value: s.by_status['Resolved'] || 0,
        icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
        bgColor: 'linear-gradient(135deg, #10b981, #059669)',
        trend: 15,
      },
    ];
  }

  private buildDeptLoad(s: Stats) {
    const deptMap: Record<string, string> = {
      'Pothole': 'Roads & Infrastructure',
      'Sinkhole': 'Roads & Infrastructure',
      'Water Leakage': 'Water & Sewage',
      'Garbage Overflow': 'Sanitation',
      'Broken Streetlight': 'Electrical',
      'Broken Sidewalk': 'Public Works',
    };
    const colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444'];
    const deptCounts: Record<string, number> = {};
    for (const [type, count] of Object.entries(s.by_type)) {
      const dept = deptMap[type] || type;
      deptCounts[dept] = (deptCounts[dept] || 0) + count;
    }
    const maxCount = Math.max(...Object.values(deptCounts), 1);
    this.departmentLoad = Object.entries(deptCounts).map(([name, count], i) => ({
      name,
      count,
      percent: (count / maxCount) * 100,
      color: colors[i % colors.length],
    }));
  }

  private calculateResolution(s: Stats) {
    const resolved = s.by_status['Resolved'] || 0;
    this.resolutionRate = s.total > 0 ? Math.round((resolved / s.total) * 100) : 0;
    const circumference = 2 * Math.PI * 50;
    const filled = (this.resolutionRate / 100) * circumference;
    this.resolutionDash = `${filled} ${circumference}`;
  }

  private buildCharts() {
    if (!this.stats) return;
    this.charts.forEach(c => c.destroy());
    this.charts = [];

    const s = this.stats;

    // Issue Type Bar Chart
    if (this.issueChartRef?.nativeElement) {
      const labels = Object.keys(s.by_type);
      const data = Object.values(s.by_type);
      this.charts.push(new Chart(this.issueChartRef.nativeElement, {
        type: 'bar',
        data: {
          labels: labels.map(l => l.replace('_', ' ')),
          datasets: [{
            data,
            backgroundColor: ['#3b82f6', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#10b981'],
            borderRadius: 6,
            borderSkipped: false,
            barThickness: 28,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } },
              border: { display: false },
            },
            y: {
              grid: { color: 'rgba(42,42,62,0.5)' },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } },
              border: { display: false },
            }
          }
        }
      }));
    }

    // Severity Donut
    if (this.severityChartRef?.nativeElement) {
      this.charts.push(new Chart(this.severityChartRef.nativeElement, {
        type: 'doughnut',
        data: {
          labels: ['High', 'Medium', 'Low'],
          datasets: [{
            data: [s.by_severity['High'] || 0, s.by_severity['Medium'] || 0, s.by_severity['Low'] || 0],
            backgroundColor: ['#ef4444', '#f59e0b', '#3b82f6'],
            borderWidth: 0,
            spacing: 3,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '65%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#8585a0', font: { size: 11, family: 'Inter' }, padding: 16, usePointStyle: true, pointStyleWidth: 8 }
            }
          }
        }
      }));
    }

    // Status Donut
    if (this.statusChartRef?.nativeElement) {
      this.charts.push(new Chart(this.statusChartRef.nativeElement, {
        type: 'doughnut',
        data: {
          labels: ['Open', 'In Progress', 'Resolved'],
          datasets: [{
            data: [s.by_status['Open'] || 0, s.by_status['In Progress'] || 0, s.by_status['Resolved'] || 0],
            backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
            borderWidth: 0,
            spacing: 3,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '65%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#8585a0', font: { size: 11, family: 'Inter' }, padding: 16, usePointStyle: true, pointStyleWidth: 8 }
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
    this.buildKpis(this.stats);
    this.buildDeptLoad(this.stats);
    this.calculateResolution(this.stats);
    setTimeout(() => this.buildCharts(), 300);
  }

  getTimeAgo(dateStr: string): string {
    const now = new Date();
    const date = new Date(dateStr);
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  selectComplaint(c: Complaint) {
    this.selectedComplaint = c;
  }

  closeModal() {
    this.selectedComplaint = null;
  }

  getImageUrl(url: string): string {
    if (url.startsWith('http')) return url;
    return `${environment.apiUrl}${url}`;
  }
}
