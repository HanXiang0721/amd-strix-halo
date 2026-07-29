#!/usr/bin/env python3
"""绘制 Test A (串行) vs Test B (并行 subagent) 时间线对比图"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1.5]})

# ============================================================
# 图1: Test A — 串行时间线
# ============================================================
ax1.set_title('Test A: 串行执行（原始提示词）', fontsize=16, fontweight='bold', pad=15)

# Test A 实际数据 (时间点, 工具, 说明)
test_a = [
    (0, 162, 'bash', '检查环境'),
    (162, 174, 'read', '读取文件'),
    (174, 183, 'bash', '检查库'),
    (183, 234, 'write', '写 filter.py'),
    (234, 268, 'bash', '测试'),
    (268, 294, 'grep×3', '验证'),
    (294, 321, 'bash', '边界测试'),
    (321, 328, 'done', '完成 OK'),
]

colors_a = {
    'bash': '#4A90D9',
    'read': '#7B68EE',
    'write': '#2ECC71',
    'grep×3': '#E67E22',
    'done': '#27AE60',
}

y_pos = 0
for start, end, tool, desc in test_a:
    color = colors_a.get(tool, '#888888')
    width = end - start
    if tool == 'done':
        ax1.barh(y_pos, width, left=start, height=0.6, color=color, alpha=0.8, edgecolor='white')
    else:
        ax1.barh(y_pos, width, left=start, height=0.6, color=color, alpha=0.8, edgecolor='white')
    # 标注
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

# 并发标注
ax1.annotate('LLM 并发数 = 1\n(始终串行)', xy=(160, 0.5), fontsize=10,
            ha='center', color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.8))

# legend
handles_a = [mpatches.Patch(color=c, label=t) for t, c in colors_a.items()]
ax1.legend(handles=handles_a, loc='upper right', fontsize=9, ncol=5)

# ============================================================
# 图2: Test B — 并行 subagent 时间线
# ============================================================
ax2.set_title('Test B: 并行 Subagent 执行（+多开subagent指令）', fontsize=16, fontweight='bold', pad=15)

# 主 agent 步骤
main_steps = [
    (0, 14, 'todowrite', '规划'),
    (14, 146, 'task×3', '启动3个subagent'),
    (150, 153, 'read', '读取结果'),
    (153, 327, 'llm', 'LLM思考(174s)'),
    (327, 341, 'write', '写 filter.py'),
    (341, 351, 'edit', '修复SVG'),
    (351, 367, 'write', '测试文件'),
    (367, 370, 'bash', '测试'),
    (370, 431, 'llm', 'LLM思考'),
    (431, 520, 'bash+edit', '检查嵌套绕过'),
    (520, 540, 'bash', '验证+超时'),
]

# subagent 步骤 (y 不同)
subagents = [
    ('subagent 1: 检查环境', 14, 32, '#3498DB', 1),
    ('subagent 2: 研究净化', 32, 87, '#9B59B6', 2),
    ('subagent 3: 研究parser', 56, 273, '#E74C3C', 3),
]

colors_b = {
    'todowrite': '#F39C12',
    'task×3': '#E74C3C',
    'read': '#7B68EE',
    'llm': '#BDC3C7',
    'write': '#2ECC71',
    'edit': '#1ABC9C',
    'bash': '#4A90D9',
    'bash+edit': '#E67E22',
}

# 画主 agent
y_main = 0
for start, end, tool, desc in main_steps:
    color = colors_b.get(tool, '#888888')
    width = end - start
    ax2.barh(y_main, width, left=start, height=0.5, color=color, alpha=0.8, edgecolor='white')
    mid = (start + end) / 2
    if width > 20:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    elif width > 8:
        ax2.text(mid, y_main, desc, ha='center', va='center', fontsize=6, color='#333')

ax2.text(-5, y_main, '主 Agent', ha='right', va='center', fontsize=10, fontweight='bold')

# 画 subagent
for name, start, end, color, y in subagents:
    width = end - start
    ax2.barh(-y, width, left=start, height=0.5, color=color, alpha=0.7, edgecolor='white', hatch='//')
    mid = (start + end) / 2
    ax2.text(mid, -y, f'{name}\n({width}s)', ha='center', va='center', fontsize=7, color='#333')
    ax2.text(-5, -y, f'Sub {y}', ha='right', va='center', fontsize=9)

# 并行窗口标注
ax2.axvspan(14, 146, alpha=0.08, color='red')
ax2.annotate('并行窗口\nbatch size = 3', xy=(80, -4), fontsize=10,
            ha='center', color='#C0392B', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', alpha=0.8))

# subagent 返回点
ax2.axvline(x=146, color='#E74C3C', linestyle='--', alpha=0.5)
ax2.text(148, 0.5, 'subagent\n全部返回', fontsize=8, color='#E74C3C', va='top')

ax2.set_xlim(-30, 560)
ax2.set_ylim(-4.5, 1)
ax2.set_xlabel('时间 (秒)', fontsize=12)
ax2.set_yticks([])
ax2.grid(axis='x', alpha=0.3)

# legend
handles_b = [mpatches.Patch(color=c, label=t) for t, c in colors_b.items()]
handles_b.append(mpatches.Patch(facecolor='#3498DB', alpha=0.7, hatch='//', label='subagent'))
ax2.legend(handles=handles_b, loc='lower right', fontsize=8, ncol=4)

plt.tight_layout()
output_path = '/home/shanhaizhibian/amd-strix-halo/docs/opencode-timeline-comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'图表已保存: {output_path}')
