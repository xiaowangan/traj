"""Migrated numerical service: ZygoReaderMixin; legacy numerical conventions retained."""
from __future__ import annotations
import struct
import numpy as np

class ZygoReaderMixin:
    @staticmethod
    def read_zygo_header(path):
        """读取 DAT 文件头里的测量波长等标定字段（不读相位数据）。

        波长位于偏移 168（单位 m，大端 float32）；示例文件均为 6.328e-7 m。只有落在
        200–2000 nm 的值才视为可信，否则返回 None，由用户手工填写。
        """
        with open(path, "rb") as stream:
            header = stream.read(4096)
        if len(header) < 238:
            raise ValueError("Zygo DAT 文件头不完整")
        wavelength_nm = struct.unpack_from(">f", header, 168)[0] * 1.0e9
        plausible = bool(np.isfinite(wavelength_nm) and 200.0 <= wavelength_nm <= 2000.0)
        return {"wavelength_nm": float(wavelength_nm) if plausible else None,
                "scale_factor": float(struct.unpack_from(">f", header, 164)[0]),
                "obliquity_factor": float(struct.unpack_from(">f", header, 176)[0])}

    @staticmethod
    def read_zygo_dat(path):
        """Read the big-endian Zygo DAT layout used by the source MLAPP."""
        with open(path, "rb") as stream:
            header = stream.read(4096)
            if len(header) < 238:
                raise ValueError("Zygo DAT 文件头不完整")

            def unpack(fmt, offset):
                return struct.unpack_from(">" + fmt, header, offset)[0]

            header_format = unpack("h", 4)
            intensity_bytes = unpack("i", 60)
            origin_x = unpack("h", 64)
            origin_y = unpack("h", 66)
            width = unpack("h", 68)
            height = unpack("h", 70)
            scale = unpack("f", 164)
            obliquity = unpack("f", 176)
            pixel_mm = unpack("f", 184) * 1000.0
            phase_code = unpack("h", 218)
            phase_resolution = 4096.0 if phase_code == 0 else 32768.0
            camera_width = unpack("h", 234)
            camera_height = unpack("h", 236)
            if min(width, height, camera_width, camera_height) <= 0:
                raise ValueError("Zygo DAT 尺寸字段无效")
            data_offset = intensity_bytes + (4096 if header_format == 3 else 834)
            stream.seek(data_offset)
            raw = stream.read(width * height * 4)
        if len(raw) != width * height * 4:
            raise ValueError("Zygo DAT 相位数据不完整")
        phase = np.frombuffer(raw, dtype=">i4").astype(float).reshape(height, width)
        phase[phase >= 2147483640] = np.nan
        phase *= scale * obliquity / phase_resolution
        full = np.full((camera_height, camera_width), np.nan, dtype=float)
        y1 = min(camera_height, origin_y + height)
        x1 = min(camera_width, origin_x + width)
        full[max(origin_y, 0):y1, max(origin_x, 0):x1] = phase[
            max(-origin_y, 0):height - max(origin_y + height - camera_height, 0),
            max(-origin_x, 0):width - max(origin_x + width - camera_width, 0),
        ]
        return full, float(pixel_mm)
