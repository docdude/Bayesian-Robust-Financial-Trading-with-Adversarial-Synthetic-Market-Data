"""Generate comparison plots for baseline vs augmented trading strategies."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])


def load_records(path):
    with open(path) as f:
        data = json.load(f)
    # Extract first element if nested lists (from vectorized env)
    result = {}
    for key in data:
        vals = data[key]
        result[key] = [v[0] if isinstance(v, list) else v for v in vals]
    return result


def parse_dates(timestamps):
    dates = []
    for t in timestamps:
        try:
            dates.append(datetime.strptime(t, '%Y-%m-%d'))
        except (ValueError, TypeError):
            try:
                dates.append(datetime.strptime(t.split('+')[0].split('T')[0], '%Y-%m-%d'))
            except Exception:
                dates.append(None)
    return dates


def plot_equity_comparison(records_dict, stage, output_path):
    """Plot portfolio value over time for multiple strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

    colors = {'Baseline (no aug)': '#d62728', 'Augmented (ckpt14)': '#2ca02c'}

    for label, records in records_dict.items():
        dates = parse_dates(records['timestamp'])
        values = [float(v) for v in records['value']]
        profits = [float(v) for v in records['total_profit']]
        color = colors.get(label, 'blue')

        # Equity curve
        axes[0].plot(dates[:len(values)], values, label=label, color=color, linewidth=1.5)

        # Cumulative profit
        axes[1].plot(dates[:len(profits)], profits, label=label, color=color, linewidth=1.5)

    axes[0].set_title(f'Portfolio Value - {stage.title()} Set', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=11)
    axes[0].axhline(y=1e7, color='gray', linestyle='--', alpha=0.5, label='Initial ($10M)')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    axes[1].set_title(f'Cumulative Profit/Loss (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Profit (%)', fontsize=11)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_actions(records, label, stage, output_path):
    """Plot price with buy/sell/hold action markers."""
    dates = parse_dates(records['timestamp'])
    prices = [float(v) for v in records['price']]
    actions = records['action']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates[:len(prices)], prices, color='#1f77b4', linewidth=1, alpha=0.7, label='Price')

    # Mark action changes
    long_dates, long_prices = [], []
    short_dates, short_prices = [], []
    close_dates, close_prices = [], []
    prev_action = None
    for i, (d, p, a) in enumerate(zip(dates, prices, actions)):
        if a != prev_action:
            if a == 'long':
                long_dates.append(d)
                long_prices.append(p)
            elif a == 'short':
                short_dates.append(d)
                short_prices.append(p)
            elif a == 'close':
                close_dates.append(d)
                close_prices.append(p)
        prev_action = a

    ax.scatter(long_dates, long_prices, marker='^', color='green', s=60, zorder=5, label='Long')
    ax.scatter(short_dates, short_prices, marker='v', color='red', s=60, zorder=5, label='Short')
    ax.scatter(close_dates, close_prices, marker='o', color='gray', s=40, zorder=5, label='Close')

    ax.set_title(f'{label} - Trading Actions ({stage.title()})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_table(metrics, output_path):
    """Create a summary metrics table as image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    headers = ['', 'Train (2000-2017)', 'Valid (2018-2020)', 'Test (2021-2024)']
    cell_text = []
    for model, data in metrics.items():
        row = [model]
        for stage in ['train', 'valid', 'test']:
            if stage in data:
                row.append(f"{data[stage]['profit']:+.1f}%  (${data[stage]['value']/1e6:.1f}M)")
            else:
                row.append('N/A')
        cell_text.append(row)

    table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color coding
    for i, row in enumerate(cell_text):
        for j in range(1, 4):
            cell = table[i + 1, j]
            text = row[j]
            if text != 'N/A' and '+' in text:
                cell.set_facecolor('#d4edda')
            elif text != 'N/A' and '-' in text.split('(')[0]:
                cell.set_facecolor('#f8d7da')

    for j in range(len(headers)):
        table[0, j].set_facecolor('#343a40')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('DBB ETF Trading Performance Comparison', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base_dir = os.path.join(ROOT, "downstream_tasks/rl/trading/workdir/exp/trading/DBB/dqn")
    fig_dir = os.path.join(ROOT, "downstream_tasks/rl/trading/workdir/fig")
    os.makedirs(fig_dir, exist_ok=True)

    # Load records
    strategies = {
        'Baseline (no aug)': os.path.join(base_dir, 'baseline_no_aug'),
        'Augmented (ckpt14)': os.path.join(base_dir, 'exp001'),
    }

    metrics = {}
    all_records = {}

    for label, path in strategies.items():
        all_records[label] = {}
        metrics[label] = {}
        for stage in ['train', 'valid', 'test']:
            fpath = os.path.join(path, f'{stage}_records.json')
            if os.path.exists(fpath):
                records = load_records(fpath)
                all_records[label][stage] = records
                values = [float(v) for v in records['value']]
                profits = [float(v) for v in records['total_profit']]
                metrics[label][stage] = {
                    'profit': profits[-1] if profits else 0,
                    'value': values[-1] if values else 0,
                }

    # 1. Summary table
    plot_summary_table(metrics, os.path.join(fig_dir, 'summary_table.png'))

    # 2. Equity comparison for each stage
    for stage in ['train', 'valid', 'test']:
        stage_records = {}
        for label in strategies:
            if stage in all_records[label]:
                stage_records[label] = all_records[label][stage]
        if stage_records:
            plot_equity_comparison(stage_records, stage, os.path.join(fig_dir, f'equity_{stage}.png'))

    # 3. Action plots for augmented model
    for stage in ['valid', 'test']:
        if stage in all_records.get('Augmented (ckpt14)', {}):
            plot_actions(all_records['Augmented (ckpt14)'][stage], 'Augmented (ckpt14)',
                         stage, os.path.join(fig_dir, f'actions_aug_{stage}.png'))
        if stage in all_records.get('Baseline (no aug)', {}):
            plot_actions(all_records['Baseline (no aug)'][stage], 'Baseline (no aug)',
                         stage, os.path.join(fig_dir, f'actions_baseline_{stage}.png'))

    print(f"\nAll plots saved to: {fig_dir}")


if __name__ == '__main__':
    main()
