import { Component, OnInit, ViewChild, ElementRef, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { Stats } from '../../models/complaint.model';
import Chart from 'chart.js/auto';

@Component({
  selector: 'app-ai-insights',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="page-container">
      <div class="page-header">
        <div>
          <h1 class="page-title">AI Model Intelligence</h1>
          <p class="page-subtitle">EfficientNetV2 + GPT-4o Vision performance metrics and detection analytics</p>
        </div>
        <div class="model-badge">
          <span class="model-dot"></span>
          EfficientNetV2-S Active
        </div>
      </div>

      <!-- Model Cards -->
      <div class="model-grid">
        <div class="model-card gradient-blue">
          <div class="mc-header">
            <div class="mc-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93L12 22"/><path d="M12 2a4 4 0 0 0-4 4c0 1.95 1.4 3.58 3.25 3.93"/><circle cx="12" cy="14" r="2"/></svg>
            </div>
            <span class="mc-label">Classification Model</span>
          </div>
          <h2 class="mc-name">EfficientNetV2-Small</h2>
          <div class="mc-specs">
            <div class="spec"><span class="spec-label">Input</span><span class="spec-value">224 x 224</span></div>
            <div class="spec"><span class="spec-label">Classes</span><span class="spec-value">6</span></div>
            <div class="spec"><span class="spec-label">Threshold</span><span class="spec-value">0.40</span></div>
            <div class="spec"><span class="spec-label">Parameters</span><span class="spec-value">~21M</span></div>
          </div>
        </div>

        <div class="model-card gradient-purple">
          <div class="mc-header">
            <div class="mc-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            </div>
            <span class="mc-label">Complaint Generator</span>
          </div>
          <h2 class="mc-name">GPT-4o Vision</h2>
          <div class="mc-specs">
            <div class="spec"><span class="spec-label">Temperature</span><span class="spec-value">0.3</span></div>
            <div class="spec"><span class="spec-label">Max Tokens</span><span class="spec-value">400</span></div>
            <div class="spec"><span class="spec-label">Fallback</span><span class="spec-value">GPT-3.5</span></div>
            <div class="spec"><span class="spec-label">Output</span><span class="spec-value">JSON</span></div>
          </div>
        </div>

        <div class="model-card gradient-cyan">
          <div class="mc-header">
            <div class="mc-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
            </div>
            <span class="mc-label">Dedup Engine</span>
          </div>
          <h2 class="mc-name">Haversine Geo</h2>
          <div class="mc-specs">
            <div class="spec"><span class="spec-label">Radius</span><span class="spec-value">50m</span></div>
            <div class="spec"><span class="spec-label">Algorithm</span><span class="spec-value">Haversine</span></div>
            <div class="spec"><span class="spec-label">Match By</span><span class="spec-value">Type + Loc</span></div>
            <div class="spec"><span class="spec-label">Skip</span><span class="spec-value">Resolved</span></div>
          </div>
        </div>
      </div>

      <!-- Detection Performance -->
      <div class="perf-grid">
        <div class="card">
          <div class="card-header">
            <h3>Detection Confidence by Class</h3>
          </div>
          <div class="chart-container">
            <canvas #confidenceChart></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h3>Classification Distribution</h3>
          </div>
          <div class="chart-container">
            <canvas #classDistChart></canvas>
          </div>
        </div>
      </div>

      <!-- Class Details -->
      <div class="card class-card">
        <div class="card-header">
          <h3>Detectable Issue Classes</h3>
          <span class="card-badge">6 Classes</span>
        </div>
        <div class="class-grid">
          <div class="class-item" *ngFor="let cls of classDetails; let i = index" [style.animation-delay]="i * 0.05 + 's'">
            <div class="class-color" [style.background]="cls.color"></div>
            <div class="class-info">
              <span class="class-name">{{ cls.name }}</span>
              <span class="class-dept">{{ cls.department }}</span>
            </div>
            <span class="badge" [ngClass]="'badge-' + cls.severity.toLowerCase()">{{ cls.severity }}</span>
            <div class="class-bar-container">
              <div class="class-bar" [style.width.%]="cls.percent" [style.background]="cls.color"></div>
            </div>
            <span class="class-count">{{ cls.count }}</span>
          </div>
        </div>
      </div>

      <!-- Pipeline Diagram -->
      <div class="card pipeline-card">
        <div class="card-header">
          <h3>AI Processing Pipeline</h3>
        </div>
        <div class="pipeline">
          <div class="pipeline-step" *ngFor="let step of pipelineSteps; let i = index; let last = last">
            <div class="step-node" [style.borderColor]="step.color">
              <span class="step-icon" [innerHTML]="step.icon"></span>
            </div>
            <div class="step-info">
              <span class="step-name">{{ step.name }}</span>
              <span class="step-desc">{{ step.desc }}</span>
            </div>
            <div class="step-arrow" *ngIf="!last">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 24px;
    }

    .model-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      background: var(--success-dim);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: var(--radius-md);
      color: var(--success);
      font-size: 12px;
      font-weight: 600;
    }

    .model-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.5s ease-in-out infinite;
    }

    /* Model Cards */
    .model-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }

    .model-card {
      border-radius: var(--radius-lg);
      padding: 24px;
      position: relative;
      overflow: hidden;

      &.gradient-blue { background: linear-gradient(135deg, #1e3a5f, #1a2744); border: 1px solid rgba(59, 130, 246, 0.3); }
      &.gradient-purple { background: linear-gradient(135deg, #2d1b4e, #1a1530); border: 1px solid rgba(139, 92, 246, 0.3); }
      &.gradient-cyan { background: linear-gradient(135deg, #0a3d4e, #0a2030); border: 1px solid rgba(6, 182, 212, 0.3); }
    }

    .mc-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }

    .mc-icon {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: rgba(255,255,255,0.1);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .mc-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: rgba(255,255,255,0.5);
      font-weight: 600;
    }

    .mc-name {
      font-size: 20px;
      font-weight: 800;
      color: white;
      margin-bottom: 16px;
      letter-spacing: -0.3px;
    }

    .mc-specs {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }

    .spec {
      display: flex;
      flex-direction: column;
    }

    .spec-label {
      font-size: 10px;
      color: rgba(255,255,255,0.4);
      text-transform: uppercase;
      letter-spacing: 0.3px;
      font-weight: 600;
    }

    .spec-value {
      font-size: 14px;
      color: rgba(255,255,255,0.9);
      font-weight: 700;
    }

    /* Performance Grid */
    .perf-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }

    .chart-container {
      height: 250px;
      position: relative;
    }

    .card-badge {
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--accent-purple-dim);
      color: var(--accent-purple);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    /* Class Details */
    .class-grid {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .class-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px solid var(--border-light);
      animation: slideUp 0.3s ease forwards;
      opacity: 0;

      &:last-child { border-bottom: none; }
    }

    .class-color {
      width: 4px;
      height: 36px;
      border-radius: 2px;
      flex-shrink: 0;
    }

    .class-info {
      flex: 1;
      display: flex;
      flex-direction: column;
    }

    .class-name {
      font-weight: 600;
      font-size: 13px;
      color: var(--text-primary);
    }

    .class-dept {
      font-size: 11px;
      color: var(--text-muted);
    }

    .class-bar-container {
      width: 120px;
      height: 5px;
      background: var(--bg-surface-2);
      border-radius: 3px;
      overflow: hidden;
    }

    .class-bar {
      height: 100%;
      border-radius: 3px;
      transition: width 1s ease;
    }

    .class-count {
      font-size: 14px;
      font-weight: 700;
      color: var(--text-primary);
      min-width: 30px;
      text-align: right;
    }

    /* Pipeline */
    .pipeline-card { overflow-x: auto; }

    .pipeline {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 20px 0;
      min-width: 800px;
    }

    .pipeline-step {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
    }

    .step-node {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      border: 2px solid;
      background: var(--bg-surface-2);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .step-info {
      display: flex;
      flex-direction: column;
      min-width: 80px;
    }

    .step-name {
      font-size: 12px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .step-desc {
      font-size: 10px;
      color: var(--text-muted);
    }

    .step-arrow {
      flex-shrink: 0;
      opacity: 0.4;
    }

    @media (max-width: 1200px) {
      .model-grid { grid-template-columns: 1fr; }
      .perf-grid { grid-template-columns: 1fr; }
    }
  `]
})
export class AiInsightsComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('confidenceChart') confidenceRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('classDistChart') classDistRef!: ElementRef<HTMLCanvasElement>;

  classDetails: any[] = [];
  stats: Stats | null = null;
  private charts: Chart[] = [];

  pipelineSteps = [
    { name: 'Upload', desc: 'Image capture', color: '#3b82f6', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>' },
    { name: 'Preprocess', desc: '224x224 + norm', color: '#8b5cf6', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>' },
    { name: 'EfficientNet', desc: 'Classification', color: '#06b6d4', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93L12 22"/><path d="M12 2a4 4 0 0 0-4 4c0 1.95 1.4 3.58 3.25 3.93"/></svg>' },
    { name: 'GPT-4o', desc: 'Complaint gen', color: '#f59e0b', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>' },
    { name: 'Dedup Check', desc: 'Haversine 50m', color: '#10b981', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>' },
    { name: 'Store', desc: 'SQLite DB', color: '#ef4444', icon: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>' },
  ];

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

  loadData() {
    this.api.getStats().subscribe({
      next: (s) => {
        this.stats = s;
        this.buildClassDetails(s);
        this.buildCharts();
      },
      error: () => this.useMockData(),
    });
  }

  private buildClassDetails(s: Stats) {
    const classInfo: Record<string, { department: string; severity: string; color: string }> = {
      'Pothole': { department: 'Roads & Infrastructure', severity: 'High', color: '#3b82f6' },
      'Sinkhole': { department: 'Roads & Infrastructure', severity: 'High', color: '#ef4444' },
      'Water Leakage': { department: 'Water & Sewage', severity: 'High', color: '#06b6d4' },
      'Garbage Overflow': { department: 'Sanitation', severity: 'Medium', color: '#f59e0b' },
      'Broken Streetlight': { department: 'Electrical', severity: 'Medium', color: '#8b5cf6' },
      'Broken Sidewalk': { department: 'Public Works', severity: 'Low', color: '#10b981' },
    };

    const maxCount = Math.max(...Object.values(s.by_type), 1);
    this.classDetails = Object.entries(classInfo).map(([name, info]) => ({
      name,
      department: info.department,
      severity: info.severity,
      color: info.color,
      count: s.by_type[name] || 0,
      percent: ((s.by_type[name] || 0) / maxCount) * 100,
    }));
  }

  private buildCharts() {
    if (!this.stats) return;
    this.charts.forEach(c => c.destroy());
    this.charts = [];

    // Confidence by class (simulated)
    if (this.confidenceRef?.nativeElement) {
      const classes = ['Pothole', 'Sinkhole', 'Water Leak', 'Garbage', 'Streetlight', 'Sidewalk'];
      const confidences = [92, 88, 85, 90, 87, 83];

      this.charts.push(new Chart(this.confidenceRef.nativeElement, {
        type: 'bar',
        data: {
          labels: classes,
          datasets: [{
            label: 'Confidence %',
            data: confidences,
            backgroundColor: ['#3b82f6', '#ef4444', '#06b6d4', '#f59e0b', '#8b5cf6', '#10b981'],
            borderRadius: 6,
            borderSkipped: false,
            barThickness: 32,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { color: '#55556a', font: { size: 10, family: 'Inter' } }, border: { display: false } },
            y: {
              min: 70, max: 100,
              grid: { color: 'rgba(42,42,62,0.5)' },
              ticks: { color: '#55556a', font: { size: 10, family: 'Inter' }, callback: (v: any) => v + '%' },
              border: { display: false },
            }
          }
        }
      }));
    }

    // Class distribution polar
    if (this.classDistRef?.nativeElement) {
      const types = Object.keys(this.stats.by_type);
      const values = Object.values(this.stats.by_type);
      this.charts.push(new Chart(this.classDistRef.nativeElement, {
        type: 'polarArea',
        data: {
          labels: types,
          datasets: [{
            data: values,
            backgroundColor: [
              'rgba(59, 130, 246, 0.6)', 'rgba(239, 68, 68, 0.6)', 'rgba(6, 182, 212, 0.6)',
              'rgba(245, 158, 11, 0.6)', 'rgba(139, 92, 246, 0.6)', 'rgba(16, 185, 129, 0.6)',
            ],
            borderWidth: 0,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
              labels: { color: '#8585a0', font: { size: 10, family: 'Inter' }, padding: 8, usePointStyle: true, pointStyleWidth: 8 }
            }
          },
          scales: {
            r: {
              grid: { color: 'rgba(42,42,62,0.5)' },
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
    this.buildClassDetails(this.stats);
    setTimeout(() => this.buildCharts(), 300);
  }
}
