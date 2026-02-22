import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  template: `
    <header class="header">
      <div class="header-left">
        <div class="breadcrumb">
          <span class="breadcrumb-root">CivicLens</span>
          <span class="breadcrumb-sep">/</span>
          <span class="breadcrumb-current">{{ pageTitle }}</span>
        </div>
      </div>
      <div class="header-right">
        <div class="header-search">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input type="text" placeholder="Search complaints..." class="search-input" />
        </div>
        <button class="header-btn notification-btn" title="Notifications">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
            <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
          </svg>
          <span class="notification-dot"></span>
        </button>
        <div class="user-avatar" title="Admin">
          <span>A</span>
        </div>
      </div>
    </header>
  `,
  styles: [`
    .header {
      height: var(--header-height);
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      position: sticky;
      top: 0;
      z-index: 40;
    }

    .breadcrumb {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }

    .breadcrumb-root { color: var(--text-muted); font-weight: 500; }
    .breadcrumb-sep { color: var(--text-muted); opacity: 0.5; }
    .breadcrumb-current { color: var(--text-primary); font-weight: 600; }

    .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .header-search {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-color);
      border-radius: var(--radius-md);
      padding: 7px 14px;
      color: var(--text-muted);
      transition: all 0.15s ease;

      &:focus-within {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px var(--accent-blue-dim);
      }
    }

    .search-input {
      background: none;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-size: 13px;
      font-family: var(--font-family);
      width: 180px;

      &::placeholder { color: var(--text-muted); }
    }

    .header-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      border: 1px solid var(--border-color);
      background: var(--bg-surface);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.15s ease;
      position: relative;

      &:hover {
        background: var(--bg-surface-hover);
        color: var(--text-primary);
      }
    }

    .notification-dot {
      position: absolute;
      top: 7px;
      right: 7px;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--danger);
      border: 2px solid var(--bg-secondary);
    }

    .user-avatar {
      width: 36px;
      height: 36px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      font-weight: 700;
      color: white;
      cursor: pointer;
    }
  `]
})
export class HeaderComponent {
  @Input() pageTitle = 'Dashboard';
}
