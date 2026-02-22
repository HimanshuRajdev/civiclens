import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  template: `
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div class="brand-icon">
          <svg viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="url(#grad)"/>
            <path d="M8 16L14 10L20 16L14 22Z" fill="white" opacity="0.9"/>
            <path d="M14 10L20 16L26 10L20 4Z" fill="white" opacity="0.6"/>
            <defs>
              <linearGradient id="grad" x1="0" y1="0" x2="32" y2="32">
                <stop stop-color="#3b82f6"/>
                <stop offset="1" stop-color="#8b5cf6"/>
              </linearGradient>
            </defs>
          </svg>
        </div>
        <div class="brand-text">
          <span class="brand-name">CivicLens</span>
          <span class="brand-badge">AI Dashboard</span>
        </div>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section">
          <span class="nav-label">Main</span>
          <a routerLink="/dashboard" routerLinkActive="active" class="nav-item">
            <span class="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
                <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
              </svg>
            </span>
            <span>Overview</span>
          </a>
          <a routerLink="/complaints" routerLinkActive="active" class="nav-item">
            <span class="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
            </span>
            <span>Complaints</span>
          </a>
          <a routerLink="/analytics" routerLinkActive="active" class="nav-item">
            <span class="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>
                <line x1="6" y1="20" x2="6" y2="14"/>
              </svg>
            </span>
            <span>Analytics</span>
          </a>
          <a routerLink="/map" routerLinkActive="active" class="nav-item">
            <span class="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>
                <line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/>
              </svg>
            </span>
            <span>Live Map</span>
          </a>
        </div>

        <div class="nav-section">
          <span class="nav-label">Intelligence</span>
          <a routerLink="/ai-insights" routerLinkActive="active" class="nav-item">
            <span class="nav-icon">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93L12 22"/>
                <path d="M12 2a4 4 0 0 0-4 4c0 1.95 1.4 3.58 3.25 3.93"/>
                <circle cx="12" cy="14" r="2"/>
              </svg>
            </span>
            <span>AI Insights</span>
            <span class="nav-badge">ML</span>
          </a>
        </div>
      </nav>

      <div class="sidebar-footer">
        <div class="system-status">
          <span class="status-dot"></span>
          <span class="status-text">System Online</span>
        </div>
        <div class="version">v1.0.0</div>
      </div>
    </aside>
  `,
  styles: [`
    .sidebar {
      width: var(--sidebar-width);
      height: 100vh;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      position: fixed;
      left: 0;
      top: 0;
      z-index: 50;
    }

    .sidebar-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 20px 20px 24px;
      border-bottom: 1px solid var(--border-light);
    }

    .brand-icon svg {
      width: 36px;
      height: 36px;
    }

    .brand-text {
      display: flex;
      flex-direction: column;
    }

    .brand-name {
      font-size: 17px;
      font-weight: 800;
      letter-spacing: -0.3px;
      color: var(--text-primary);
    }

    .brand-badge {
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--accent-blue);
      opacity: 0.8;
    }

    .sidebar-nav {
      flex: 1;
      padding: 16px 12px;
      overflow-y: auto;
    }

    .nav-section {
      margin-bottom: 24px;
    }

    .nav-label {
      display: block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
      padding: 0 12px;
      margin-bottom: 8px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: var(--radius-md);
      color: var(--text-secondary);
      text-decoration: none;
      font-size: 13.5px;
      font-weight: 500;
      transition: all 0.15s ease;
      margin-bottom: 2px;
      position: relative;

      &:hover {
        background: var(--bg-surface);
        color: var(--text-primary);
      }

      &.active {
        background: var(--accent-blue-dim);
        color: var(--accent-blue);

        .nav-icon { color: var(--accent-blue); }

        &::before {
          content: '';
          position: absolute;
          left: 0;
          top: 50%;
          transform: translateY(-50%);
          width: 3px;
          height: 20px;
          background: var(--accent-blue);
          border-radius: 0 3px 3px 0;
        }
      }
    }

    .nav-icon {
      display: flex;
      align-items: center;
      color: var(--text-muted);
      transition: color 0.15s ease;
    }

    .nav-badge {
      margin-left: auto;
      font-size: 9px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: var(--accent-purple-dim);
      color: var(--accent-purple);
      letter-spacing: 0.5px;
    }

    .sidebar-footer {
      padding: 16px 20px;
      border-top: 1px solid var(--border-light);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .system-status {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
      animation: pulse 2s ease-in-out infinite;
    }

    .status-text {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
    }

    .version {
      font-size: 10px;
      color: var(--text-muted);
    }
  `]
})
export class SidebarComponent {}
