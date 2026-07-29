#!/usr/bin/env python3
"""llm-inference-batching-scheduler: Test A (串行) vs Test B (background subagent 并行)"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 1.3]})

# ============================================================
# 图1: Test A — 串行 (12步, 680s, 完成)
# ============================================================
ax1.set_title('Test A: 串行执行 — llm-inference-batching-scheduler (12步, 680s, 完成)', 
              fontsize=13, fontweight='bold', pad=15)

phases_a = [
    (0, 468, '#BDC3C7', 'LLM思考+规划 (468s)'),
    (468, 512, '#4A90D9', 'bash+todowrite+write'),
    (512, 580, '#2ECC71', 'write cost_model.py'),
    (580, 596, '#2ECC71', 'write scheduler.py'),
    (596, 638, '#4A90D9', 'bash 运行+验证'),
    (638, 665, '#E67E22', 'bash 基准测试'),
    (665, 680, '#27AE60', '完成'),
]

y = 0
for start, end, color, desc in phases_a:
    width = end - start
    ax1.barh(y, width, left=start, height=0.6, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 35:
        ax1.text(mid, y, desc, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax1.axvline(x=680, color='#27AE60', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(682, 0.4, '完成 680s', fontsize=9, color='#27AE60', fontweight='bold')

ax1.set_xlim(-10, 720)
ax1.set_ylim(-0.8, 0.8)
ax1.set_xlabel('时间 (秒)', fontsize=12)
ax1.set_yticks([])
ax1.grid(axis='x', alpha=0.3)

ax1.annotate('LLM 并发 = 1\n12步串行\n6个文件全部生成', xy=(200, -0.5), fontsize=9,
            ha='center', color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.8))

handles_a = [
    mpatches.Patch(color='#BDC3C7', label='LLM 思考'),
    mpatches.Patch(color='#4A90D9', label='bash/todowrite'),
    mpatches.Patch(color='#2ECC71', label='write'),
    mpatches.Patch(color='#E67E22', label='bash 测试'),
    mpatches.Patch(color='#27AE60', label='完成'),
]
ax1.legend(handles=handles_a, loc='upper right', fontsize=8, ncol=5)

# ============================================================
# 图2: Test B — Background Subagent 并行 (5步主agent, 4 subagent, 274s)
# ============================================================
ax2.set_title('Test B: Background Subagent 并行 — llm-inference-batching-scheduler (5步主agent + 4 background subagent)', 
              fontsize=13, fontweight='bold', pad=15)

# 主 agent
main_b = [
    (0, 209, '#BDC3C7', 'LLM思考 (209s)'),
    (209, 257, '#F39C12', 'todowrite+task×4'),
    (257, 262, '#4A90D9', 'bash (检查文件)'),
    (262, 274, '#27AE60', '等待+完成'),
]

y_main = 0
for start, end, color, desc in main_b:
    width = end - start
    ax2.barh(y_main, width, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 15:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=7, color='white', fontweight='bold')

ax2.text(-5, y_main, '主 Agent', ha='right', va='center', fontsize=10, fontweight='bold')

# 4 个 background subagent (从 t=257s 启动)
sub_start = 257
# Sub 1: 数据生成 (t=262s 完成, ~5s)
ax2.barh(-1, 5, left=sub_start, height=0.5, color='#3498DB', alpha=0.7, edgecolor='white', hatch='//')
ax2.text(sub_start + 10, -1, 'Sub 1: 数据生成 (5s, 完成)', ha='left', va='center', fontsize=7, color='#333')

# Sub 2: 数据生成 (t=262s 完成, ~5s)
ax2.barh(-2, 5, left=sub_start, height=0.5, color='#9B59B6', alpha=0.7, edgecolor='white', hatch='//')
ax2.text(sub_start + 10, -2, 'Sub 2: 数据生成 (5s, 完成)', ha='left', va='center', fontsize=7, color='#333')

# Sub 3: cost_model.py (运行中, 主agent退出时未完成)
ax2.barh(-3, 274-sub_start, left=sub_start, height=0.5, color='#E74C3C', alpha=0.5, edgecolor='white', hatch='//')
ax2.text(sub_start + 10, -3, 'Sub 3: cost_model.py (运行中, 被终止)', ha='left', va='center', fontsize=7, color='#333')

# Sub 4: scheduler.py (运行中, 主agent退出时未完成)
ax2.barh(-4, 274-sub_start, left=sub_start, height=0.5, color='#E67E22', alpha=0.5, edgecolor='white', hatch='//')
ax2.text(sub_start + 10, -4, 'Sub 4: scheduler.py (运行中, 被终止)', ha='left', va='center', fontsize=7, color='#333')

for i in range(1, 5):
    ax2.text(-5, -i, f'Sub {i}', ha='right', va='center', fontsize=9)

# 并行窗口
ax2.axvspan(sub_start, 262, alpha=0.1, color='green')
ax2.annotate('4个subagent\n同时运行\nbatch size = 4', xy=(259, -4.8), fontsize=9,
            ha='center', color='#27AE60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

# task 启动
ax2.axvline(x=sub_start, color='#E74C3C', linestyle=':', alpha=0.5)
ax2.text(sub_start + 1, 0.4, 'background\ntask×4启动', fontsize=7, color='#E74C3C')

# 主 agent 退出
ax2.axvline(x=274, color='#7F8C8D', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(276, 0.4, '主agent退出\nsubagent被终止', fontsize=8, color='#7F8C8D')

ax2.set_xlim(-30, 310)
ax2.set_ylim(-5.5, 0.8)
ax2.set_xlabel('时间 (秒)', fontsize=12)
ax2.set_yticks([])
ax2.grid(axis='x', alpha=0.3)

handles_b = [
    mpatches.Patch(color='#BDC3C7', label='LLM 思考'),
    mpatches.Patch(color='#F39C12', label='todowrite + task(background)'),
    mpatches.Patch(color='#4A90D9', label='bash'),
    mpatches.Patch(color='#27AE60', label='主 agent 完成'),
    mpatches.Patch(facecolor='#3498DB', alpha=0.7, hatch='//', label='background subagent (完成)'),
    mpatches.Patch(facecolor='#E74C3C', alpha=0.5, hatch='//', label='background subagent (被终止)'),
]
ax2.legend(handles=handles_b, loc='lower right', fontsize=7, ncol=3)

plt.tight_layout()
output = '/home/shanhaizhibian/amd-strix-halo/docs/opencode-batching-scheduler-timeline.png'
plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
print(f'图表已保存: {output}')
