#!/usr/bin/env python3
"""Test A (串行) vs Test B (background subagent 并行) 时间线对比"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 1.2]})

# ============================================================
# 图1: Test A — 串行
# ============================================================
ax1.set_title('Test A: 串行执行（无 background subagent）', fontsize=14, fontweight='bold', pad=15)

phases_a = [
    (0, 104, '#BDC3C7', 'LLM思考 (104s)'),
    (104, 142, '#4A90D9', 'bash+todowrite'),
    (142, 161, '#2ECC71', 'write filter.py'),
    (161, 210, '#4A90D9', 'bash+read 测试'),
    (210, 245, '#E67E22', 'read+edit 修复'),
    (245, 300, '#4A90D9', 'bash+write 测试 (超时)'),
]

y = 0
for start, end, color, desc in phases_a:
    width = end - start
    ax1.barh(y, width, left=start, height=0.6, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 30:
        ax1.text(mid, y, desc, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax1.axvline(x=300, color='#C0392B', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(302, 0.4, '超时 300s', fontsize=9, color='#C0392B', fontweight='bold')

ax1.set_xlim(-10, 330)
ax1.set_ylim(-0.8, 0.8)
ax1.set_xlabel('时间 (秒)', fontsize=12)
ax1.set_yticks([])
ax1.grid(axis='x', alpha=0.3)

ax1.annotate('LLM 并发 = 1\n14步全串行\nfilter.py 已生成', xy=(120, -0.5), fontsize=9,
            ha='center', color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.8))

handles_a = [
    mpatches.Patch(color='#BDC3C7', label='LLM 思考'),
    mpatches.Patch(color='#4A90D9', label='bash/read'),
    mpatches.Patch(color='#2ECC71', label='write'),
    mpatches.Patch(color='#E67E22', label='edit'),
]
ax1.legend(handles=handles_a, loc='upper right', fontsize=8, ncol=4)

# ============================================================
# 图2: Test B — Background Subagent 并行
# ============================================================
ax2.set_title('Test B: Background Subagent 并行（OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true）', 
              fontsize=14, fontweight='bold', pad=15)

# 主 agent
main_b = [
    (0, 21, '#BDC3C7', 'LLM思考 (21s)'),
    (21, 43, '#F39C12', 'todowrite+task×2'),
    (43, 58, '#2ECC71', 'write (测试文件)'),
    (58, 70, '#27AE60', '主agent完成'),
]

y_main = 0
for start, end, color, desc in main_b:
    width = end - start
    ax2.barh(y_main, width, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 10:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=7, color='white', fontweight='bold')

ax2.text(-5, y_main, '主 Agent', ha='right', va='center', fontsize=10, fontweight='bold')

# Background subagent (从 task 启动后开始, 不确定精确结束时间, 画到 300s)
sub_start = 43
ax2.barh(-1, 300-sub_start, left=sub_start, height=0.5, color='#3498DB', alpha=0.5, 
         edgecolor='white', hatch='//')
ax2.text(170, -1, 'Sub 1: Explore environment (background, 主agent退出后仍运行)', 
         ha='center', va='center', fontsize=7, color='#333')

ax2.barh(-2, 300-sub_start, left=sub_start, height=0.5, color='#9B59B6', alpha=0.5,
         edgecolor='white', hatch='//')
ax2.text(170, -2, 'Sub 2: Research sanitization (background, 主agent退出后仍运行)', 
         ha='center', va='center', fontsize=7, color='#333')

ax2.text(-5, -1, 'Sub 1', ha='right', va='center', fontsize=9)
ax2.text(-5, -2, 'Sub 2', ha='right', va='center', fontsize=9)

# 并行标注
ax2.axvspan(43, 70, alpha=0.1, color='green')
ax2.annotate('主agent与subagent\n同时运行\nbatch size = 3', xy=(56, -3.2), fontsize=9,
            ha='center', color='#27AE60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

# 主 agent 退出点
ax2.axvline(x=70, color='#27AE60', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(72, 0.4, '主agent完成\nsubagent继续', fontsize=8, color='#27AE60')

# task 启动点
ax2.axvline(x=43, color='#E74C3C', linestyle=':', alpha=0.5)
ax2.text(44, -2.5, 'background\ntask启动\n(不等返回)', fontsize=7, color='#E74C3C')

ax2.set_xlim(-30, 320)
ax2.set_ylim(-3.5, 0.8)
ax2.set_xlabel('时间 (秒)', fontsize=12)
ax2.set_yticks([])
ax2.grid(axis='x', alpha=0.3)

handles_b = [
    mpatches.Patch(color='#BDC3C7', label='LLM 思考'),
    mpatches.Patch(color='#F39C12', label='todowrite + task(background)'),
    mpatches.Patch(color='#2ECC71', label='write'),
    mpatches.Patch(color='#27AE60', label='主 agent 完成'),
    mpatches.Patch(facecolor='#3498DB', alpha=0.5, hatch='//', label='background subagent'),
]
ax2.legend(handles=handles_b, loc='lower right', fontsize=8, ncol=3)

plt.tight_layout()
output = '/home/shanhaizhibian/amd-strix-halo/docs/opencode-background-subagent-timeline.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f'图表已保存: {output}')
