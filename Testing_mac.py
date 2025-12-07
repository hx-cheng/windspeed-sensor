import sys, time, collections, os, re, csv
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import serial, serial.tools.list_ports
try:
    import yaml
except ImportError:
    yaml = None

class SerialReader(QtCore.QThread):
    data_received = QtCore.Signal(str, float, float)  # port, ts, value
    disconnected = QtCore.Signal(str)  # port

    def __init__(self, port_name: str, baud: int = 115200, parent=None):
        super().__init__(parent)
        self._port_name = port_name
        self._baud = baud
        self._stop = False
        self._last_data_ts = 0.0

    def run(self):
        try:
            ser = serial.Serial(self._port_name, self._baud, timeout=1)
        except Exception as e:
            print(f"Failed to open {self._port_name}: {e}")
            try:
                self.disconnected.emit(self._port_name)
            except Exception:
                pass
            return
        # Capture start time for relative timestamps
        start_time = time.monotonic()
        with ser:
            while not self._stop:
                try:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        # If we haven't seen data in a while and the device disappeared, treat as disconnect
                        now = time.monotonic()
                        if (now - self._last_data_ts) > 2.0:
                            try:
                                ports = [p.device for p in serial.tools.list_ports.comports() if "usbmodem" in p.device]
                                grp = _modem_group(self._port_name)
                                present = any(_modem_group(d) == grp for d in ports)
                                if not present:
                                    print(f"Port disappeared: {self._port_name}")
                                    break
                            except Exception:
                                pass
                        continue
                    # Expect "ID,value" or just "value"
                    parts = line.split(",")
                    value_str = parts[-1]
                    value = float(value_str)
                    # Use monotonic time for high-resolution, stable timestamps
                    ts = time.monotonic()
                    self.data_received.emit(self._port_name, ts, value)
                    self._last_data_ts = ts
                except ValueError:
                    continue
                except Exception as e:
                    print(f"Read error on {self._port_name}: {e}")
                    break
        try:
            self.disconnected.emit(self._port_name)
        except Exception:
            pass

    def stop(self):
        self._stop = True
        self.wait(500)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, devices, max_points=3000, refresh_ms=30, window_seconds=15, max_speed=10.0, min_speed=0.0):
        super().__init__()
        self.setWindowTitle("NUCLEO Live Plot")
        self.resize(900, 500)

        # Layout: top numeric readouts, bottom live plot
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Controls bar: logging toggle and status
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 0)
        controls_layout.setSpacing(12)
        self.btn_log = QtWidgets.QPushButton("Start Logging")
        self.lbl_log_status = QtWidgets.QLabel("Logging: OFF")
        self.lbl_log_status.setStyleSheet("color: #666;")
        controls_layout.addWidget(self.btn_log)
        controls_layout.addWidget(self.lbl_log_status)
        controls_layout.addStretch(1)
        main_layout.addWidget(controls)

        self.header = QtWidgets.QWidget()
        self.header_layout = QtWidgets.QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(8, 8, 8, 8)
        self.header_layout.setSpacing(16)
        main_layout.addWidget(self.header)

        self.plot = pg.PlotWidget()
        main_layout.addWidget(self.plot, 1)
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # Control x-range manually for rolling window; fix y-range to avoid outliers
        try:
            self.plot.getViewBox().enableAutoRange(x=False, y=False)
            self.plot.setLabel('left', 'Speed', units='m/s')
            self.plot.setLabel('bottom', 'Time', units='s')
        except Exception:
            pass

        self.max_points = max_points
        self.buffers = {}
        self.curves = {}
        self.t0 = time.monotonic()  # Use monotonic clock for stable timing
        self.value_labels = {}
        self.port_to_reader = {}
        self.port_to_widget = {}
        self.window_seconds = float(window_seconds)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.port_to_name = {}
        self.is_ground_truth_port = {}
        self.logging_active = False
        self.log_rows_gt = []
        self.log_rows_student = []
        # Error line state
        self.err_lower = {}
        self.err_upper = {}
        self.err_fill = {}
        self.y_log = False
        self._log_mode_applied = False
        self.air_density = 1.204
        self.dp_error_pa = 0.62
        self.error_speed_floor = 0.2

        self.palette = ['c', 'm', 'y', 'g', 'r', 'b', 'w']
        self.next_color_idx = 0
        self.readers = []
        self.known_ports = set()
        self.ground_truth_port = None
        self.ground_truth_group = None
        self.student_port = None

        for idx, dev in enumerate(devices):
            port = dev.get('port')
            name = dev.get('name') or port
            baud = int(dev.get('baud', 115200))
            if not port:
                continue

            # buffers per port
            self.buffers[port] = (
                collections.deque(maxlen=self.max_points),
                collections.deque(maxlen=self.max_points),
            )

            # curve per port
            color = self.palette[self.next_color_idx % len(self.palette)]
            self.next_color_idx += 1
            label = f"{name} ({port})" if name and name != port else port

            # numeric readout per device
            device_widget = QtWidgets.QWidget()
            device_layout = QtWidgets.QVBoxLayout(device_widget)
            device_layout.setContentsMargins(8, 4, 8, 4)
            title_label = QtWidgets.QLabel(label)
            title_label.setStyleSheet("color: #666; font-size: 12pt;")
            value_label = QtWidgets.QLabel("—")
            value_label.setStyleSheet("font-size: 24pt; font-weight: 600;")
            device_layout.addWidget(title_label)
            device_layout.addWidget(value_label)
            self.header_layout.addWidget(device_widget)
            self.value_labels[port] = value_label
            self.curves[port] = self.plot.plot(pen=pg.mkPen(color, width=2), name=label)
            self.port_to_name[port] = name
            # Mark ground truth by name if provided, else fallback later
            if isinstance(name, str) and name.strip().lower().startswith('ground truth'):
                self.is_ground_truth_port[port] = True
            else:
                self.is_ground_truth_port.setdefault(port, False)

            # error lines for this device
            base_pen = self.curves[port].opts['pen']
            qcolor = base_pen.color()
            dash_pen = pg.mkPen(qcolor, style=QtCore.Qt.PenStyle.DashLine)
            self.err_lower[port] = self.plot.plot(pen=dash_pen)
            self.err_upper[port] = self.plot.plot(pen=dash_pen)
            brush = pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 60)
            fbi = pg.FillBetweenItem(self.err_lower[port], self.err_upper[port], brush=brush)
            self.err_fill[port] = fbi
            self.plot.addItem(fbi)

            # serial reader per port
            reader = SerialReader(port, baud=baud)
            reader.data_received.connect(self.on_data)
            reader.disconnected.connect(self.on_disconnected)
            reader.start()
            self.readers.append(reader)
            self.known_ports.add(port)
            self.port_to_reader[port] = reader

            # Initialize ground truth tracking from the first configured device
            if self.ground_truth_port is None:
                self.ground_truth_port = port
                self.ground_truth_group = _modem_group(port)
                # If not explicitly named, treat the first as ground truth
                if not self.is_ground_truth_port.get(port, False):
                    self.is_ground_truth_port[port] = True
            else:
                # Any subsequent attached device at init is considered student
                self.student_port = port

        # Apply fixed y-range now that max_speed is known
        try:
            self.plot.setYRange(self.min_speed, self.max_speed, padding=0)
        except Exception:
            pass

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(refresh_ms))  # ~33 FPS by default

        # Hot-plug auto-detect: attach a second device when it appears
        self.desired_device_count = 2
        if len(self.curves) < self.desired_device_count:
            print("Waiting for additional /dev/tty.usbmodem* device to attach...")
        self.scan_timer = QtCore.QTimer(self)
        self.scan_timer.timeout.connect(self.scan_and_attach_new_devices)
        self.scan_timer.start(5000)

        # Hook up logging toggle
        self.btn_log.clicked.connect(self.toggle_logging)

    @QtCore.Slot(str, float, float)
    def on_data(self, port, ts, value):
        t_rel = ts - self.t0  # ts is already from monotonic clock
        # Clip incoming values to keep scale stable
        clipped = value
        try:
            if clipped < self.min_speed:
                clipped = self.min_speed
            if clipped > self.max_speed:
                clipped = self.max_speed
        except Exception:
            pass
        tbuf, vbuf = self.buffers[port]
        tbuf.append(t_rel)
        vbuf.append(clipped)
        lbl = self.value_labels.get(port)
        if lbl is not None:
            try:
                lbl.setText(f"{clipped:.3f} m/s")
            except Exception:
                pass
        # Logging rows: ISO time, epoch, t_rel, device_name, port, raw, clipped, expected_error_pct, cov_lower_mps, cov_upper_mps
        if self.logging_active:
            try:
                # Convert monotonic time to wall clock time for ISO timestamp
                wall_time = time.time()
                iso = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(wall_time)) + f".{int((wall_time%1)*1000):03d}Z"
                rho = float(getattr(self, 'air_density', 1.204))
                dp = float(getattr(self, 'dp_error_pa', 0.62))
                vabs = abs(float(clipped))
                # Only compute error for ground truth device
                # Prefer explicit marking by name; fallback to group match
                is_gt = bool(self.is_ground_truth_port.get(port, False)) or (_modem_group(port) == getattr(self, 'ground_truth_group', None))
                if is_gt and vabs > 1e-12:
                    dv_abs = dp / (rho * vabs)
                    err_pct = (dv_abs / vabs) * 100.0
                    err_pct_str = f"{err_pct:.3f}"
                    cov_lo = max(self.min_speed if not getattr(self, 'y_log', False) else max(self.min_speed, 1e-6), float(clipped) - dv_abs)
                    cov_hi = float(clipped) + dv_abs
                    cov_lo_str = f"{cov_lo:.6f}"
                    cov_hi_str = f"{cov_hi:.6f}"
                else:
                    err_pct_str = ''
                    cov_lo_str = ''
                    cov_hi_str = ''
                row = [
                    iso,
                    f"{ts:.6f}",
                    f"{t_rel:.6f}",
                    self.port_to_name.get(port, port) or port,
                    port,
                    f"{float(value):.6f}",
                    f"{float(clipped):.6f}",
                    err_pct_str,
                    cov_lo_str,
                    cov_hi_str,
                ]
                if is_gt:
                    self.log_rows_gt.append(row)
                else:
                    self.log_rows_student.append(row)
            except Exception:
                pass

    def refresh(self):
        # Apply log mode if requested
        if not self._log_mode_applied and getattr(self, 'y_log', False):
            if self.min_speed <= 0:
                self.min_speed = 1e-6
            try:
                self.plot.setLogMode(y=True)
                self._log_mode_applied = True
            except Exception:
                pass

        latest_ts = None
        # Update curves and error lines; find latest timestamp
        for port, curve in self.curves.items():
            tbuf, vbuf = self.buffers[port]
            if not tbuf:
                continue
            t_list = list(tbuf)
            v_list = list(vbuf)
            curve.setData(t_list, v_list)

            # dp-based error: dv = dp / (rho * max(|v|, floor))
            rho = float(getattr(self, 'air_density', 1.204))
            dp = float(getattr(self, 'dp_error_pa', 0.62))
            v_floor = float(getattr(self, 'error_speed_floor', 0.2))
            lower = []
            upper = []
            min_floor = self.min_speed if not getattr(self, 'y_log', False) else max(self.min_speed, 1e-6)
            for v in v_list:
                denom = rho * max(abs(v), v_floor)
                dv = (dp / denom) if denom > 0 else 0.0
                lo = v - dv
                hi = v + dv
                if lo < min_floor:
                    lo = min_floor
                lower.append(lo)
                upper.append(hi)
            self.err_lower[port].setData(t_list, lower)
            self.err_upper[port].setData(t_list, upper)

            if latest_ts is None or t_list[-1] > latest_ts:
                latest_ts = t_list[-1]

        if latest_ts is not None:
            start = max(0.0, latest_ts - self.window_seconds)
            try:
                self.plot.setXRange(start, latest_ts, padding=0)
            except Exception:
                pass
        # Dynamic Y-axis: fit to current window max + 1, honor min_speed/log
        try:
            global_max = None
            if latest_ts is not None:
                t_cut = latest_ts - self.window_seconds
            else:
                t_cut = None
            for port in self.curves.keys():
                tbuf, vbuf = self.buffers[port]
                if not tbuf:
                    continue
                if t_cut is None:
                    local_max = max(vbuf)
                else:
                    local_max = None
                    for t, v in zip(tbuf, vbuf):
                        if t >= t_cut:
                            local_max = v if local_max is None else (v if v > local_max else local_max)
                if local_max is not None:
                    global_max = local_max if global_max is None else (local_max if local_max > global_max else global_max)
            if global_max is not None:
                upper = global_max + 1.0
                lower = self.min_speed if not getattr(self, 'y_log', False) else max(self.min_speed, 1e-6)
                if upper <= lower:
                    upper = lower * 2.0
                self.plot.setYRange(lower, upper, padding=0)
        except Exception:
            pass

    def closeEvent(self, event):
        for r in self.readers:
            r.stop()
        super().closeEvent(event)

    def attach_device(self, port: str, name: str = 'student', baud: int = 115200):
        if port in self.curves:
            return
        # buffers
        self.buffers[port] = (
            collections.deque(maxlen=self.max_points),
            collections.deque(maxlen=self.max_points),
        )
        # UI elements
        color = self.palette[self.next_color_idx % len(self.palette)]
        self.next_color_idx += 1
        label = f"{name} ({port})" if name and name != port else port
        device_widget = QtWidgets.QWidget()
        device_layout = QtWidgets.QVBoxLayout(device_widget)
        device_layout.setContentsMargins(8, 4, 8, 4)
        title_label = QtWidgets.QLabel(label)
        title_label.setStyleSheet("color: #666; font-size: 12pt;")
        value_label = QtWidgets.QLabel("—")
        value_label.setStyleSheet("font-size: 24pt; font-weight: 600;")
        device_layout.addWidget(title_label)
        device_layout.addWidget(value_label)
        self.header_layout.addWidget(device_widget)
        self.value_labels[port] = value_label
        self.port_to_widget[port] = device_widget
        self.curves[port] = self.plot.plot(pen=pg.mkPen(color, width=2), name=label)
        self.port_to_name[port] = name
        # Error lines
        base_pen = self.curves[port].opts['pen']
        qcolor = base_pen.color()
        dash_pen = pg.mkPen(qcolor, style=QtCore.Qt.PenStyle.DashLine)
        self.err_lower[port] = self.plot.plot(pen=dash_pen)
        self.err_upper[port] = self.plot.plot(pen=dash_pen)
        brush = pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 60)
        fbi = pg.FillBetweenItem(self.err_lower[port], self.err_upper[port], brush=brush)
        self.err_fill[port] = fbi
        self.plot.addItem(fbi)
        # Reader
        reader = SerialReader(port, baud=baud)
        reader.data_received.connect(self.on_data)
        reader.disconnected.connect(self.on_disconnected)
        reader.start()
        self.readers.append(reader)
        self.known_ports.add(port)
        self.port_to_reader[port] = reader
        print(f"Attached new device: {label} @ {baud}")

        # Track student assignment if not ground truth
        if _modem_group(port) != self.ground_truth_group:
            if self.student_port and self.student_port in self.curves and self.student_port != port:
                self.remove_device(self.student_port)
            self.student_port = port

    @QtCore.Slot()
    def scan_and_attach_new_devices(self):
        try:
            if len(self.curves) >= self.desired_device_count:
                return
            known_groups = {_modem_group(p) for p in self.known_ports}
            available = [p for p in _list_available_device_paths() if p not in self.known_ports and _modem_group(p) not in known_groups]
            if available:
                # Attach the first new usbmodem device we see
                self.attach_device(available[0], name='student', baud=115200)
        except Exception as e:
            # Avoid crashing the UI if scan fails
            print(f"Scan error: {e}")

    @QtCore.Slot(str)
    def on_disconnected(self, port: str):
        try:
            if port not in self.curves:
                return
            print(f"Device disconnected: {port}")
            self.remove_device(port)
            if self.student_port == port:
                self.student_port = None
        except Exception as e:
            print(f"Error handling disconnect for {port}: {e}")

    def remove_device(self, port: str):
        # Stop and remove reader
        reader = self.port_to_reader.pop(port, None)
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
        # Remove plot curve
        curve = self.curves.pop(port, None)
        if curve is not None:
            try:
                self.plot.removeItem(curve)
            except Exception:
                pass
        # Remove error items
        lower = self.err_lower.pop(port, None)
        upper = self.err_upper.pop(port, None)
        fill = self.err_fill.pop(port, None)
        for item in (lower, upper, fill):
            if item is not None:
                try:
                    self.plot.removeItem(item)
                except Exception:
                    pass
        # Remove header widget
        widget = self.port_to_widget.pop(port, None)
        if widget is not None:
            try:
                widget.setParent(None)
                widget.deleteLater()
            except Exception:
                pass
        # Remove buffers and labels
        self.buffers.pop(port, None)
        self.value_labels.pop(port, None)
        if port in self.known_ports:
            self.known_ports.remove(port)
        self.port_to_name.pop(port, None)

    @QtCore.Slot()
    def toggle_logging(self):
        if not self.logging_active:
            # Start logging
            self.logging_active = True
            self.log_rows_gt = []
            self.log_rows_student = []
            self.btn_log.setText("Stop && Save")
            self.lbl_log_status.setText("Logging: ON")
            self.lbl_log_status.setStyleSheet("color: #0a0;")
        else:
            # Stop logging and prompt to save
            self.logging_active = False
            self.btn_log.setText("Start Logging")
            self.lbl_log_status.setText("Logging: OFF")
            self.lbl_log_status.setStyleSheet("color: #666;")
            if not self.log_rows_gt and not self.log_rows_student:
                return
            try:
                # Save Ground Truth file if present
                if self.log_rows_gt:
                    path_gt, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Ground Truth CSV", "", "CSV Files (*.csv)")
                    if path_gt:
                        if not path_gt.lower().endswith('.csv'):
                            path_gt += '.csv'
                        self._save_log_csv(path_gt, self.log_rows_gt)
                # Save Student file if present
                if self.log_rows_student:
                    path_stu, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Student CSV", "", "CSV Files (*.csv)")
                    if path_stu:
                        if not path_stu.lower().endswith('.csv'):
                            path_stu += '.csv'
                        self._save_log_csv(path_stu, self.log_rows_student)
            except Exception as e:
                print(f"Failed to save CSV: {e}")

    def _save_log_csv(self, filepath: str, rows):
        # Simple approach: save all readings with their actual timestamps
        # No forced alignment - do that in post-processing
        header = [
            'timestamp_iso', 'timestamp_epoch', 't_rel_s',
            'device_name', 'port', 'value_raw_mps', 'value_clipped_mps', 'expected_error_pct',
            'cov_lower_mps', 'cov_upper_mps'
        ]
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"Saved log to {filepath} ({len(rows)} rows)")
        except Exception as e:
            print(f"Error writing CSV {filepath}: {e}")

def list_ports():
    print("Available serial ports:")
    for p in serial.tools.list_ports.comports():
        print(f" - {p.device}: {p.manufacturer} {p.product} SN={p.serial_number}")

def _modem_group(path: str):
    m = re.search(r"usbmodem(.*)$", path)
    return m.group(1) if m else path

def _list_available_device_paths():
    ports = list(serial.tools.list_ports.comports())
    # Consider both tty.usbmodem* and cu.usbmodem*
    groups = {}
    for p in ports:
        dev = p.device
        if "usbmodem" not in dev:
            continue
        g = _modem_group(dev)
        entry = groups.setdefault(g, {"tty": None, "cu": None})
        if "/tty." in dev:
            entry["tty"] = dev
        elif "/cu." in dev:
            entry["cu"] = dev
    # Prefer tty if present, otherwise cu
    preferred = []
    for g, entry in groups.items():
        preferred.append(entry["tty"] or entry["cu"])
    preferred = [p for p in preferred if p]
    preferred.sort()
    return preferred

def _resolve_devices_from_config(config_devices):
    # Normalize and auto-detect missing ports
    all_ports = _list_available_device_paths()
    specified_ports = [d.get('port') for d in config_devices if d.get('port')]
    specified_groups = {_modem_group(p) for p in specified_ports}
    remaining = [p for p in all_ports if p not in specified_ports and _modem_group(p) not in specified_groups]

    resolved = []
    for d in (config_devices or []):
        name = d.get('name') or d.get('port') or 'device'
        baud = int(d.get('baud', 115200))
        port = d.get('port')
        if not port:
            port = remaining.pop(0) if remaining else None
        if port:
            resolved.append({'name': name, 'port': port, 'baud': baud})

    # If only one device configured (e.g., ground truth) and another port exists, add it as student
    if len(resolved) < 2 and len(all_ports) >= 2:
        used = {r['port'] for r in resolved}
        used_groups = {_modem_group(p) for p in used}
        extras = [p for p in all_ports if p not in used and _modem_group(p) not in used_groups]
        if extras:
            resolved.append({'name': 'student', 'port': extras[0], 'baud': 115200})

    return resolved

if __name__ == "__main__":
    if yaml is None:
        print("PyYAML is required. Install with: python3 -m pip install pyyaml")
        sys.exit(1)

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        list_ports()
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to read config.yaml: {e}")
        sys.exit(1)

    devices = config.get("devices", []) or []
    devices = _resolve_devices_from_config(devices)
    print("Resolved devices:")
    for d in devices:
        try:
            print(f" - {d['name']}: {d['port']} @ {d['baud']}")
        except Exception:
            pass
    ui_cfg = config.get("ui", {}) or {}
    max_points = int(ui_cfg.get("max_points", 3000))
    refresh_ms = int(ui_cfg.get("refresh_ms", 30))
    window_seconds = float(ui_cfg.get("window_seconds", 15))
    max_speed = float(ui_cfg.get("max_speed", 10))
    min_speed = float(ui_cfg.get("min_speed", 0))
    log_rate_hz = float(ui_cfg.get("log_rate_hz", 20))
    align_tolerance_ms = float(ui_cfg.get("align_tolerance_ms", 40))
    y_log = bool(ui_cfg.get("y_log", False))
    air_density = float(ui_cfg.get("air_density", 1.204))
    dp_error_pa = float(ui_cfg.get("dp_error_pa", 0.62))
    error_speed_floor = float(ui_cfg.get("error_speed_floor", 0.2))

    if not devices:
        print("No devices configured in config.yaml under 'devices'.")
        list_ports()
        sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(devices, max_points=max_points, refresh_ms=refresh_ms, window_seconds=window_seconds, max_speed=max_speed, min_speed=min_speed)
    w.log_rate_hz = log_rate_hz
    w.align_tolerance_ms = align_tolerance_ms
    w.y_log = y_log
    w.air_density = air_density
    w.dp_error_pa = dp_error_pa
    w.error_speed_floor = error_speed_floor
    w.show()
    sys.exit(app.exec())