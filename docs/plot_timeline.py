#!/usr/bin/env python3
"""绘制 Test A (串行) vs Test B (并行 subagent) 时间线对比图"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1.5]})

# ============================================================
# 图1: Test A — 串行时间线
# ============================================================
ax1.set_title('Test A: 串行执行（原始提示词）', fontsize=16, fontweight='bold', pad=15)

test_a = [
    (0, 162, '#4A90D9', 'bash', '检查环境'),
    (162, 174, '#7B68EE', 'read', '读取文件'),
    (174, 183, '#4A90D9', 'bash', '检查库'),
    (183, 234, '#2ECC71', 'write', '写 filter.py'),
    (234, 268, '#4A90D9', 'bash', '测试'),
    (268, 294, '#E67E22', 'grep', '验证'),
    (294, 321, '#4A90D9', 'bash', '边界测试'),
    (321, 328, '#27AE60', 'done', '完成'),
]

y_pos = 0
for start, end, color, tool, desc in test_a:
    width = end - start
    ax1.barh(y_pos, width, left=start, height=0.6, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 15:
        ax1.text(mid, y_pos, desc, ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    else:
        ax1.text(end + 5, y_pos, desc, ha='left', va='center', fontsize=8, color='#333')

ax1.set_xlim(-10, 350)
ax1.set_ylim(-0.8, 0.8)
ax1.set_xlabel('时间 (秒)', fontsize=12)
ax1.set_yticks([])
ax1.axhline(y=0, color='#ddd', linewidth=0.5)
ax1.grid(axis='x', alpha=0.3)

ax1.annotate('LLM 并发数 = 1\n(始终串行)', xy=(160, 0.55), fontsize=10,
            ha='center', color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.8))

colors_a = ['#4A90D9', '#7B68EE', '#2ECC71', '#E67E22', '#27AE60']
labels_a = ['bash', 'read', 'write', 'grep', 'done']
handles_a = [mpatches.Patch(color=c, label=l) for c, l in zip(colors_a, labels_a)]
ax1.legend(handles=handles_a, loc='upper right', fontsize=9, ncol=5)

# ============================================================
# 图2: Test B — 并行 subagent 时间线
# ============================================================
ax2.set_title('Test B: 并行 Subagent 执行（+多开subagent指令）', fontsize=16, fontweight='bold', pad=15)

# 主 agent 步骤 (基于实际时间戳，基准 t=0)
main_steps = [
    (0, 14, '#F39C12', '规划'),
    (14, 146, '#E74C3C', '启动subagent'),
    (150, 153, '#7B68EE', 'read'),
    (153, 327, '#BDC3C7', 'LLM思考(174s)'),
    (327, 341, '#2ECC71', '写 filter.py'),
    (341, 351, '#1ABC9C', 'edit'),
    (351, 367, '#2ECC71', '测试文件'),
    (367, 370, '#4A90D9', '测试'),
    (370, 431, '#BDC3C7', 'LLM思考'),
    (431, 520, '#E67E22', '检查绕过'),
    (520, 540, '#4A90D9', '验证+超时'),
]

# Subagent 实际时间 (相对于 t=0 = 1785242578000)
# sub1: t=18~36s  sub2: t=32~88s  sub3: t=56~273s
subagents = [
    ('Sub 1: 检查环境', 18, 36, '#3498DB', 1),
    ('Sub 2: 研究净化', 32, 88, '#9B59B6', 2),
    ('Sub 3: 研究parser', 56, 273, '#E74C3C', 3),
]

# 画主 agent
y_main = 0
for start, end, color, desc in main_steps:
    width = end - start
    ax2.barh(y_main, width, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 25:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    elif width > 8:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=6, color='#333')

ax2.text(-5, y_main, '主 Agent', ha='right', va='center', fontsize=10, fontweight='bold')

# 画 subagent
for name, start, end, color, y in subagents:
    width = end - start
    ax2.barh(-y, width, left=start, height=0.5, color=color, alpha=0.7, edgecolor='white', hatch='//')
    mid = (start + end) / 2
    label = f'{name}\n({width}s)'
    if width > 30:
        ax2.text(mid, -y, label, ha='center', va='center', fontsize=7, color='#333')
    else:
        ax2.text(end + 5, -y, label, ha='left', va='center', fontsize=7, color='#333')
    ax2.text(-5, -y, f'Sub {y}', ha='right', va='center', fontsize=9)

# 并行窗口标注 (实际并行区域)
ax2.axvspan(18, 88, alpha=0.08, color='green')
ax2.annotate('2-3个subagent\n同时运行\nbatch size = 2~3', xy=(50, -4.2), fontsize=9,
            ha='center', color='#27AE60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

# subagent 3 仍在运行时的主 agent 状态
ax2.axvline(x=146, color='#7F8C8D', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.annotate('主agent收到\ntask结果\n(但Sub3仍在跑)', xy=(146, 0.4), fontsize=8,
            ha='center', color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ECF0F1', alpha=0.8))

# Sub3 完成点
ax2.axvline(x=273, color='#E74C3C', linestyle=':', alpha=0.6, linewidth=1.5)
ax2.text(275, -3.5, 'Sub3完成', fontsize=8, color='#E74C3C', va='top')

# Sub3 运行期间主 agent 的活动
ax2.annotate('', xy=(273, -0.4), xytext=(273, -3),
            arrowprops=dict(arrowstyle='<->', color='#E74C3C', alpha=0.5))
ax2.text(310, -1.5, 'Sub3 运行期间\n主agent已在执行', fontsize=7, color='#E74C3C', ha='center')

ax2.set_xlim(-30, 560)
ax2.set_ylim(-4.5, 1)
ax2.set_xlabel('时间 (秒)', fontsize=12)
ax2.set_yticks([])
ax2.grid(axis='x', alpha=0.3)

handles_b = [
    mpatches.Patch(color='#F39C12', label='todowrite'),
    mpatches.Patch(color='#E74C3C', label='task(subagent)'),
    mpatches.Patch(color='#7B68EE', label='read'),
    mpatches.Patch(color='#BDC3C7', label='LLM 思考'),
    mpatches.Patch(color='#2ECC71', label='write'),
    mpatches.Patch(color='#1ABC9C', label='edit'),
    mpatches.Patch(color='#4A90D9', label='bash'),
    mpatches.Patch(color='#E67E22', label='bash+edit'),
    mpatches.Patch(facecolor='#3498DB', alpha=0.7, hatch='//', label='subagent 运行'),
]
ax2.legend(handles=handles_b, loc='lower right', fontsize=7, ncol=4)

plt.tight_layout()
output_path = '/home/shanhaizhibian/amd-strix-halo/docs/opencode-timeline-comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'图表已保存: {output_path}')
