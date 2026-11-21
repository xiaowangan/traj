


import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel
from PyQt5.QtCore import Qt


def _downsample_2d(data, max_side=1536):
    """大图按整数步长抽稀，避免超大干涉图拖慢 matplotlib 渲染。"""
    data = np.asarray(data, dtype=float)
    stride = 1
    while max(data.shape) / stride > max_side:
        stride += 1
    return data if stride == 1 else data[::stride, ::stride]


def _configure_chinese_font():
    """Register installed Windows CJK fonts so Matplotlib does not fall back to tofu."""
    from matplotlib import font_manager, rcParams
    names = []
    for path in (r"C:\Windows\Fonts\msyh.ttc",
                 r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        if not Path(path).is_file():
            continue
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            if name not in names:
                names.append(name)
        except Exception:
            continue
    rcParams["font.sans-serif"] = names + ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

def _wrap_cjk(text, width):
    """中文没有空格，matplotlib 的自动折行会在奇怪的位置断开，这里按字数手动换行。"""
    lines = []
    for line in text.split("\n"):
        while len(line) > width:
            lines.append(line[:width])
            line = "  " + line[width:]
        lines.append(line)
    return "\n".join(lines)


class SurfaceDataPreview(QWidget):
    """独立 matplotlib 热力图面板：面形/抛光斑页用 1×2 两面板，
    驻留时间求解页用 2×2 四面板（上下各两个），颜色样式保持一致。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #dfe9f5;")
        self._error = None
        self._colorbars = []
        self._axes = []
        self._canvases = []
        self._panel_hosts = []
        self._panel_count = 2
        # 四面板（驻留时间求解）大小调节：左/右/下三边留白像素数，
        # 数值越大四个图越小；顶部留白由横条避让自动决定，不在此处。
        # 两面板页面不受此设置影响。
        self._four_panel_inset = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # 2×2 网格：两面板模式只显示第一行，四面板模式全部显示
        self._panels = QGridLayout()
        self._panels.setContentsMargins(0, 0, 0, 0)
        self._panels.setSpacing(2)

        try:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            _configure_chinese_font()
            for index in range(4):
                host = QWidget()
                host_layout = QVBoxLayout(host)
                host_layout.setContentsMargins(0, 0, 0, 0)
                host.plot_layout = host_layout
                figure = Figure(figsize=(3.6, 3.6), dpi=100)
                figure.patch.set_facecolor("#dfe9f5")
                canvas = FigureCanvasQTAgg(figure)
                canvas.setStyleSheet("background-color: #dfe9f5;")
                axis = figure.add_subplot(111)
                axis.set_facecolor("#dfe9f5")
                figure.subplots_adjust(
                    left=0.03, right=0.80, top=0.97, bottom=0.03)
                host_layout.addWidget(canvas)
                self._axes.append(axis)
                self._canvases.append(canvas)
                self._panel_hosts.append(host)
                self._panels.addWidget(host, index // 2, index % 2, 1, 1)
                self._panels.setColumnStretch(index % 2, 1)
                self._panels.setRowStretch(index // 2, 1)
        except Exception as exc:
            self._error = exc
            hint = QLabel(
                "二维图显示需要 matplotlib。请安装后重启软件。\n\n" + str(exc))
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#c0392b; font-size:14px;")
            outer.addWidget(hint)
            return

        outer.addLayout(self._panels, 1)
        self.set_panel_count(2)

    def set_panel_count(self, count):
        """两面板（1×2）与四面板（2×2）布局切换，多余面板隐藏。"""
        count = 4 if count >= 3 else 2
        self._panel_count = count
        for index, host in enumerate(self._panel_hosts):
            host.setVisible(index < count)
            # 切回两面板时复位绘图区上边距（四面板的下移只属于求解页）
            if count == 2:
                plot_layout = getattr(host, "plot_layout", None)
                if plot_layout is not None and plot_layout.contentsMargins().top() != 0:
                    plot_layout.setContentsMargins(0, 0, 0, 0)
        # 隐藏部件不会取消其所在行的拉伸因子：两面板时必须把第二行拉伸清零，
        # 否则空行仍占据一半高度，两个面板会被挤到上半部分
        self._panels.setRowStretch(1, 1 if count == 4 else 0)
        # 四面板大小调节：只给求解页四面板加边距，两面板恢复贴边
        inset = self._four_panel_inset if count == 4 else 0
        self._panels.setContentsMargins(inset, 0, inset, inset)

    def set_panels(self, datasets, colorbar_label="面形误差 (nm)"):
        if self._error is not None:
            return
        self.set_panel_count(len(datasets))
        self._line_mode = False
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._colorbars = []
        try:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap("jet").copy()
            cmap.set_bad("#d6e2f0")
        except Exception:
            cmap = "jet"
        for ax, data in zip(self._axes, datasets):
            ax.clear()
            ax.figure.subplots_adjust(left=0.03, right=0.80, top=0.97, bottom=0.03)
            ax.set_facecolor("#dfe9f5")
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(_downsample_2d(data), cmap=cmap, aspect="equal",
                           interpolation="nearest", origin="lower")
            cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
            if colorbar_label:
                cb.set_label(colorbar_label, color="#10243f", fontsize=9)
            cb.ax.tick_params(labelsize=8, colors="#10243f")
            cb.outline.set_edgecolor("#10243f")
            self._colorbars.append(cb)
        for canvas in self._canvases:
            canvas.draw_idle()

    def set_data(self, raw, processed, colorbar_label="面形误差 (nm)"):
        self.set_panels((raw, processed), colorbar_label)

    def set_evaluation(self, result):
        """Show the latest evaluation spectra and a compact metrics summary."""
        if self._error is not None:
            return
        _configure_chinese_font()
        self.set_panel_count(4)
        self._line_mode = True
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._colorbars = []
        foreground = "#10243f"
        background = "#dfe9f5"
        for ax in self._axes:
            ax.clear()
            ax.set_facecolor(background)
            ax.figure.patch.set_facecolor(background)
            ax.figure.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
            ax.tick_params(labelsize=8, colors=foreground)

        # 2-D spatial spectrum; keep the established surface-preview color map.
        psd = np.asarray(result["psd"], dtype=float)
        positive = psd[np.isfinite(psd) & (psd > 0)]
        if positive.size:
            vmin = float(np.min(positive))
            vmax = float(np.max(positive))
            if vmax <= vmin:
                vmax = vmin * 10.0
        else:
            vmin, vmax = 1e-30, 1.0
        from matplotlib.colors import LogNorm
        image = self._axes[0].imshow(
            np.ma.masked_less_equal(psd, 0), origin="lower",
            aspect="auto", extent=(result["qx"][0], result["qx"][-1],
                                    result["qy"][0], result["qy"][-1]),
            cmap="jet", norm=LogNorm(vmin=vmin, vmax=vmax))
        self._axes[0].set_xlabel("空间频率 f_x (周期/mm)", fontsize=8, color=foreground)
        self._axes[0].set_ylabel("空间频率 f_y (周期/mm)", fontsize=8, color=foreground)
        unit = result.get("height_unit", "未知高度单位")
        cb = self._axes[0].figure.colorbar(image, ax=self._axes[0], fraction=0.05, pad=0.04)
        cb.set_label(f"PSD ({unit}²·mm²，对数色标)", fontsize=8, color=foreground)
        cb.ax.tick_params(labelsize=7, colors=foreground)
        cb.outline.set_edgecolor(foreground)
        self._colorbars.append(cb)

        freq = np.asarray(result["radial_frequency"])
        power = np.asarray(result["radial_psd"])
        valid = np.isfinite(freq) & np.isfinite(power) & (freq > 0) & (power > 0)
        if np.any(valid):
            self._axes[1].loglog(freq[valid], power[valid], color="#245b91", linewidth=1.5)
            self._axes[1].set_xlim(float(np.min(freq[valid])), float(np.max(freq[valid])))
        self._axes[1].set_aspect("auto")
        self._axes[1].axvspan(result["band_min_mm_inv"], result["band_max_mm_inv"],
                              color="#d49b48", alpha=0.22)
        self._axes[1].set_xlabel("空间频率 (周期/mm)", fontsize=8, color=foreground)
        self._axes[1].set_ylabel(f"PSD ({unit}²·mm²)", fontsize=8, color=foreground)
        self._axes[1].grid(True, which="both", alpha=0.2, color="#52657a")

        self._axes[2].bar(result["direction_centers_deg"], result["direction_rms"],
                          width=8, color="#587f9e")
        self._axes[2].set_aspect("auto")
        self._axes[2].set_xlabel("空间频率方向 (°)", fontsize=8, color=foreground)
        self._axes[2].set_ylabel(f"带内 RMS ({unit})", fontsize=8, color=foreground)
        self._axes[2].grid(True, axis="y", alpha=0.2, color="#52657a")

        y0, y1, x0, x1 = result["roi"]
        stats = result["stats"]
        detrended = result["detrended_stats"]
        lines = [
            f"检测区域：{result['roi_shape'][1]} × {result['roi_shape'][0]} 点",
            f"采样间距：{result['dx_mm']:g} × {result['dy_mm']:g} mm",
            f"有效率：{result['valid_fraction']:.1%}",
            f"PV：{stats['pv']:.5g} {unit}",
            f"RMS：{stats['rms']:.5g} {unit}",
            f"去低阶 RMS：{detrended['rms']:.5g} {unit}",
            f"中频带：{result['band_min_mm_inv']:g}–{result['band_max_mm_inv']:g} 周期/mm",
            f"中频 RMS：{result['band_rms']:.5g} {unit}",
            f"低/高频 RMS：{result['low_frequency_rms']:.4g} / "
            f"{result['high_frequency_rms']:.4g} {unit}",
            f"主导空间频率方向：{result['dominant_direction_deg']:.0f}°",
            f"ROI 索引：X[{x0}:{x1}]，Y[{y0}:{y1}]",
        ]
        self._axes[3].figure.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.18)
        self._axes[3].set_axis_off()
        self._axes[3].text(0.5, 0.5, "\n".join(lines), transform=self._axes[3].transAxes,
                           va="center", ha="center", fontsize=9, color=foreground,
                           linespacing=1.55, multialignment="center")
        for canvas in self._canvases:
            canvas.draw_idle()

    def set_assessment(self, assessment):
        """加工前/预测后/实测后的检测对比；共形振动分支则额外显示中频 RMS 随时间的演化。"""
        if self._error is not None:
            return
        from polishing.core.assessment import format_comparison
        _configure_chinese_font()
        self.set_panel_count(4)
        self._line_mode = True
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._colorbars = []
        foreground, background = "#10243f", "#dfe9f5"
        for ax in self._axes:
            ax.clear()
            ax.set_facecolor(background)
            ax.set_aspect("auto")
            ax.figure.patch.set_facecolor(background)
            ax.figure.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.18)
            ax.tick_params(labelsize=8, colors=foreground)
        unit = assessment.get("unit", "")
        states = assessment["states"]
        styles = (("before", "加工前", "#245b91"), ("predicted", "预测后", "#c0392b"),
                  ("measured", "实测后", "#2e8b57"))
        present = [(key, label, color) for key, label, color in styles if states.get(key)]
        band = assessment["protocol"]

        spectrum = self._axes[0]
        for key, label, color in present:
            freq = np.asarray(states[key]["radial_frequency"], dtype=float)
            power = np.array([np.nan if v is None else v for v in states[key]["radial_psd"]], dtype=float)
            keep = np.isfinite(power) & (power > 0) & (freq > 0)
            if np.any(keep):
                spectrum.loglog(freq[keep], power[keep], color=color, linewidth=1.5, label=label)
        spectrum.axvspan(band["band_min"], band["band_max"], color="#d49b48", alpha=0.22)
        spectrum.set_xlabel("空间频率 (周期/mm)", fontsize=8, color=foreground)
        spectrum.set_ylabel(f"PSD ({unit}²·mm²)", fontsize=8, color=foreground)
        spectrum.grid(True, which="both", alpha=0.2, color="#52657a")
        spectrum.legend(fontsize=8, loc="best")

        trend = self._axes[1]
        if assessment["kind"] == "conformal_vibration":
            times = np.asarray(assessment["times_s"], dtype=float)
            trend.plot(times, assessment["msf_rms_curve"], color="#c0392b", linewidth=1.6, label="预测中频 RMS")
            limit = (assessment.get("target") or {}).get("msf_rms_max")
            if limit is not None:
                trend.axhline(limit, color="#2e8b57", linestyle="--", linewidth=1, label=f"目标 {limit:g}")
            trend.plot([assessment["best_time_s"]], [assessment["best_msf_rms"]], "o",
                       color="#10243f", label=f"最佳 {assessment['best_time_s']:.0f} s")
            trend.set_xlabel("加工时间 (s)", fontsize=8, color=foreground)
            trend.set_ylabel(f"中频 RMS ({unit})", fontsize=8, color=foreground)
            trend.legend(fontsize=8, loc="best")
        else:
            width = 0.8 / max(len(present), 1)
            for index, (key, label, color) in enumerate(present):
                values = [states[key]["low_rms"], states[key]["msf_rms"], states[key]["high_rms"]]
                trend.bar(np.arange(3) + index * width, values, width=width, color=color, label=label)
            trend.set_xticks(np.arange(3) + width * (len(present) - 1) / 2)
            trend.set_xticklabels(["低频", "中频", "高频"], fontsize=8)
            trend.set_ylabel(f"分带 RMS ({unit})", fontsize=8, color=foreground)
            trend.legend(fontsize=8, loc="best")
        trend.grid(True, axis="y", alpha=0.2, color="#52657a")

        direction = self._axes[2]
        centers = np.asarray(states["before"]["direction_centers_deg"], dtype=float)
        width = 8.0 / max(len(present), 1)
        for index, (key, label, color) in enumerate(present):
            direction.bar(centers + index * width, states[key]["direction_rms"], width=width,
                          color=color, label=label)
        direction.set_xlabel("空间频率方向 (°)", fontsize=8, color=foreground)
        direction.set_ylabel(f"中频带内 RMS ({unit})", fontsize=8, color=foreground)
        direction.grid(True, axis="y", alpha=0.2, color="#52657a")
        direction.legend(fontsize=8, loc="best")

        summary = self._axes[3]
        summary.figure.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
        summary.set_axis_off()
        # 雅黑没有 ✔ ✘ ⚠ 字形，绘图文字里换成字体里有的符号，避免出现方块
        plain = (format_comparison(assessment).replace("｜", "   ").replace("✔", "√")
                 .replace("✘", "×").replace("⚠", "注意："))
        summary.text(0.5, 0.5, _wrap_cjk(plain, 38),
                     transform=summary.transAxes, va="center", ha="center", fontsize=8,
                     color=foreground, linespacing=1.5, multialignment="left")
        for canvas in self._canvases:
            canvas.draw_idle()

    def set_fit_curves(self, result):
        if self._error is not None:
            return
        self.set_panel_count(2)
        from matplotlib import rcParams
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
        self._line_mode = True
        for cb in self._colorbars:
            cb.remove()
        self._colorbars = []
        curves = (("x", "gap", 0.05, "接触间隙 (mm)"),
                  ("rate_x", "rate", 5e-5, "空间变化率 |Δs/Δx| (mm/mm)"))
        for ax, (xkey, ykey, threshold, ylabel) in zip(self._axes, curves):
            ax.clear(); ax.set_facecolor("#dfe9f5")
            ax.set_aspect("auto")
            ax.figure.subplots_adjust(left=0.19, right=0.96, top=0.94, bottom=0.17)
            if result is not None:
                ax.plot(result[xkey], result[ykey], color="#245b91", linewidth=1.6)
                ax.set_xlim(0, result["x"][-1])
                ymax = max(threshold * 1.4, float(np.max(result[ykey])) * 1.15)
            else:
                ymax = threshold * 1.4
                ax.text(0.5, 0.5, "填写参数并点击抛光盘口径校验", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#52657a")
            ax.axhline(threshold, color="#c0392b", linestyle="--", linewidth=1,
                       label=f"参考上限 {threshold:g}")
            ax.set_ylim(0, ymax)
            ax.set_xlabel("工件母线位置 x (mm)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(labelsize=9, colors="#10243f")
            ax.grid(True, alpha=0.25); ax.legend(loc="upper right", fontsize=9)
            ax.figure.canvas.draw_idle()
