# -*- coding: utf-8 -*-
import hashlib, hmac, json, os, datetime, platform

_SECRET_KEY  = b"PolishingSoftware_PlanarTraj_2026"
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LICENSE_FILE = os.path.join(_BASE_DIR, "license.dat")


def get_hardware_id():
    if platform.system() == "Windows":
        try:
            import wmi
            c = wmi.WMI()
            raw = f"{c.Win32_Processor()[0].ProcessorId.strip()}|{c.Win32_BaseBoard()[0].SerialNumber.strip()}"
        except Exception as e:
            raw = f"WMI_ERR|{str(e)[:32]}"
    else:
        import socket
        user = os.environ.get("USER", os.environ.get("USERNAME", "dev"))
        try: user = os.getlogin()
        except: pass
        raw = f"{socket.gethostname()}|{user}"
    d = hashlib.sha256(raw.encode()).hexdigest().upper()
    return "-".join(d[i*6:(i+1)*6] for i in range(4))


def generate_activation_code(hwid, valid_days):
    mac = hmac.new(_SECRET_KEY, f"{hwid}|{valid_days}".encode(), hashlib.sha256).hexdigest().upper()
    return "-".join(mac[:32][i*8:(i+1)*8] for i in range(4))


def activate(input_code, valid_days):
    hwid = get_hardware_id()
    expected = generate_activation_code(hwid, valid_days)
    norm = lambda s: s.strip().upper().replace("-","").replace(" ","")
    if norm(input_code) != norm(expected):
        return False, "激活码错误，请检查后重新输入"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"hwid": hwid, "valid_days": valid_days,
            "first_activation": now, "last_run": now}
    _write(data)
    expiry = (datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
              + datetime.timedelta(days=valid_days)).strftime("%Y-%m-%d")
    return True, f"激活成功！有效期 {valid_days} 天，到期时间：{expiry}"


def verify_license():
    if not os.path.exists(LICENSE_FILE):
        return False, "未找到授权文件，请激活软件"
    data = _read()
    if data is None:
        return False, "授权文件已损坏或被篡改，请重新激活"
    if data.get("hwid") != get_hardware_id():
        return False, "机器码不匹配，授权文件与当前设备不符"
    expected = generate_activation_code(data["hwid"], data["valid_days"])
    if data.get("code_hash") != hashlib.sha256(expected.encode()).hexdigest():
        return False, "授权文件完整性校验失败"
    try:
        first  = datetime.datetime.strptime(data["first_activation"], "%Y-%m-%d %H:%M:%S")
        last   = datetime.datetime.strptime(data["last_run"],          "%Y-%m-%d %H:%M:%S")
        now    = datetime.datetime.now()
        if now < last - datetime.timedelta(minutes=5):
            return False, "检测到系统时间异常（时钟回拨）"
        expiry = first + datetime.timedelta(days=data["valid_days"])
        if now > expiry:
            return False, f"授权已过期（{expiry.strftime('%Y-%m-%d')}），请重新激活"
        data["last_run"] = now.strftime("%Y-%m-%d %H:%M:%S")
        _write(data)
        return True, f"授权有效，剩余 {(expiry-now).days} 天（到期：{expiry.strftime('%Y-%m-%d')}）"
    except Exception as e:
        return False, f"验证异常：{e}"


def _write(data):
    code = generate_activation_code(data["hwid"], data["valid_days"])
    data["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
    content = json.dumps(data, ensure_ascii=False, indent=2)
    mac = hmac.new(_SECRET_KEY, content.encode(), hashlib.sha256).hexdigest()
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        f.write(content + "\n---\n" + mac)


def _read():
    try:
        text = open(LICENSE_FILE, encoding="utf-8").read()
        parts = text.split("\n---\n")
        if len(parts) != 2: return None
        content, stored = parts
        if not hmac.compare_digest(stored.strip(),
               hmac.new(_SECRET_KEY, content.encode(), hashlib.sha256).hexdigest()):
            return None
        return json.loads(content)
    except: return None
