# ============================================================
# MODULE 1 — ModelState
# ============================================================
# Shared state for BOTH 2D and 3D interactive diagrams.
# All transformations, angles, distances, positions, and
# path evaluations depend on this central, synchronized state.
# ============================================================
# ============================================================
# Codex.py — Fully self-contained, single-file version
# ============================================================
import sys
import math
import numpy as np

from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QColor

# Remove ALL modular imports like:
#   from main_window import MainWindow
#   from .model_state import ModelState
#   from .diagram_2d import Diagram2DWidget
#   from .diagram_3d import Diagram3DWidget
#   etc.
#
# The entire file now contains all modules inline.



class ModelState(QtCore.QObject):
    """
    Centralized shared state for the entire Codex diagram system.

    You selected OPTION 1:
        - Second point is ALWAYS derived from (r2, theta2)
        - Not draggable
        - z2 is manually controlled (3D elevation only)

    Coordinate conventions:
    -------------------------------------------------------------
    2D Mode:
        Base line: x ∈ [0, U], y = 0
        Pivot    : P1 = (u, 0)
        Second   : P2 = (u + r2*cos(theta2),
                         r2*sin(theta2))

    3D Mode:
        Square base: (sx, sy) ∈ [-1,1]^2
        Pivot    : P1 = (sx, sy, 0)
        Second   : P2 = (sx + r2*cos(theta2),
                         sy + r2*sin(theta2),
                         z2)

        Rendering uses:
            world_x = sx * R
            world_y = sy * R
            world_z = z

    All state mutations emit:  changed = QtCore.pyqtSignal()
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, U: float = 12.0, parent=None):
        super().__init__(parent)

        # --------------------------------------------
        # 1D LINE STATE
        # --------------------------------------------
        self._U = float(U) if U > 0 else 1.0
        self._u = self._U / 2.0     # pivot along segment

        # --------------------------------------------
        # 2D + 3D PRIMARY POINT (pivot)
        # --------------------------------------------
        # normalized square coordinate (sx, sy) ∈ [-1,1]^2
        # sx derived from u; sy = 0 at start
        self._sx = 0.0
        self._sy = 0.0

        # --------------------------------------------
        # 2D + 3D SECOND POINT PARAMETERS
        # --------------------------------------------
        self._theta2 = 0.0  # direction from pivot
        self._r2 = 0.5      # distance
        self._z2 = 0.0      # elevation in 3D

        # --------------------------------------------
        # Derived polar coords for pivot (used by 2D/3D UI)
        # --------------------------------------------
        self._r_norm = 0.0
        self._theta = 0.0

        # Sync initial relations
        self._sync_from_sx_sy()

    # ============================================================
    # INTERNAL SYNC METHODS
    # ============================================================

    def _sync_from_sx_sy(self):
        """
        Update:
            - r_norm
            - theta
            - u from sx
        """

        # Polar coords inside square plane
        r = math.hypot(self._sx, self._sy)
        self._r_norm = min(1.0, r)
        self._theta = math.atan2(self._sy, self._sx) if r > 1e-12 else 0.0

        # Convert square-x coordinate to 1D segment coordinate
        t = (self._sx + 1.0) / 2.0     # normalized [0,1]
        t = max(0.0, min(1.0, t))
        self._u = t * self._U

    # ============================================================
    # PROPERTIES — READ ONLY
    # ============================================================

    @property
    def U(self):       return self._U
    @property
    def u(self):       return self._u
    @property
    def sx(self):      return self._sx
    @property
    def sy(self):      return self._sy

    @property
    def theta(self):   return self._theta     # polar angle of pivot
    @property
    def r_norm(self):  return self._r_norm

    @property
    def theta2(self):  return self._theta2    # angle of second point
    @property
    def r2(self):      return self._r2
    @property
    def z2(self):      return self._z2

    # ============================================================
    # PUBLIC MUTATORS — EMIT STATE CHANGES
    # ============================================================

    def set_U(self, U: float):
        U = float(U)
        if U <= 0: U = 1.0
        if abs(U - self._U) < 1e-12:
            return

        self._U = U

        # remap u to new scale
        self._u = max(0.0, min(self._u, self._U))

        # map u → sx (primary square coordinate)
        t = self._u / self._U
        self._sx = 2.0 * t - 1.0

        self._sync_from_sx_sy()
        self.changed.emit()

    def set_u_from_line(self, u: float):
        u = float(u)
        u = max(0.0, min(u, self._U))
        self._u = u

        # convert to square coordinate
        t = self._u / self._U if self._U > 0 else 0.0
        self._sx = 2.0 * t - 1.0
        self._sy = 0.0    # line always y=0 in 2D

        self._sync_from_sx_sy()
        self.changed.emit()

    def set_from_square(self, sx: float, sy: float):
        sx = max(-1.0, min(1.0, float(sx)))
        sy = max(-1.0, min(1.0, float(sy)))

        self._sx = sx
        self._sy = sy

        self._sync_from_sx_sy()
        self.changed.emit()

    # --------------------------------------------
    # Second point angle, distance, elevation
    # --------------------------------------------
    def set_theta2(self, theta2: float):
        self._theta2 = float(theta2)
        self.changed.emit()

    def set_r2(self, r2: float):
        r2 = float(r2)
        r2 = max(0.0, min(10.0, r2))  # r2 is unbounded mathematically, but clamped for UI
        self._r2 = r2
        self.changed.emit()

    def set_z2(self, z2: float):
        self._z2 = float(z2)
        self.changed.emit()

    # ============================================================
    # DERIVED COORDINATES FOR SECOND POINT
    # ============================================================

    def get_second_point_2d(self):
        """
        Returns (x2, y2) for the 2D diagram.
        Always derived from pivot using (r2, theta2).
        """
        x1 = self._u
        y1 = 0.0
        x2 = x1 + self._r2 * math.cos(self._theta2)
        y2 = y1 + self._r2 * math.sin(self._theta2)
        return x2, y2

    def get_second_point_3d_square(self):
        """
        Returns (sx2, sy2, z2) for the 3D diagram (square plane).
        """
        sx2 = self._sx + self._r2 * math.cos(self._theta2)
        sy2 = self._sy + self._r2 * math.sin(self._theta2)
        return sx2, sy2, self._z2

# ============================================================
# MODULE 2 — Diagram2DWidget
# ============================================================
# The 2D diagram contains:
#   • A horizontal line segment [0, U]
#   • A draggable pivot point P1 = (u, 0)
#   • A projective ray from P1 at angle theta2
#   • A second point P2 derived from (r2, theta2)
#   • An angle arc marking theta2
# ============================================================



class DraggablePoint(pg.ScatterPlotItem):
    """
    Draggable pivot point used in the 2D diagram.
    Dragging this updates the ModelState.u value.
    """

    def __init__(self, state: ModelState, view_box: pg.ViewBox):
        super().__init__(
            symbol='o',
            size=14,
            brush='w',
            pen=pg.mkPen(0, 0, 0, width=1)
        )
        self.state = state
        self.vb = view_box

    # --------------------------------------------
    # Handle drag events
    # --------------------------------------------
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            ev.accept()
            return
        if ev.isFinish():
            ev.accept()
            return

        ev.accept()

        pos_view = self.vb.mapSceneToView(ev.scenePos())
        x = float(pos_view.x())

        # Clamp to [0, U]
        U = self.state.U
        x = max(0.0, min(U, x))

        self.state.set_u_from_line(x)


class Diagram2DWidget(QtWidgets.QWidget):
    """
    The 2D interactive diagram with:
        - Horizontal line segment [0, U]
        - Draggable pivot
        - Projective ray (theta2)
        - Second point P2
        - Angle arc indicator
    """

    def __init__(self, state: ModelState, parent=None):
        super().__init__(parent)
        self.state = state
        self._updating = False

        # --------------------------------------------------------
        # LAYOUT
        # --------------------------------------------------------
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --------------------------------------------------------
        # PLOT WIDGET (pyqtgraph)
        # --------------------------------------------------------
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot, 1)

        self.plot.setBackground('k')
        self.plot.showGrid(x=False, y=False)
        self.plot.showAxis('left', False)
        self.plot.showAxis('bottom', False)
        self.plot.setMouseEnabled(x=False, y=False)

        vb = self.plot.getViewBox()
        vb.setAspectLocked(False)

        # --------------------------------------------------------
        # SEGMENT ITEM
        # --------------------------------------------------------
        self.segment_item = pg.PlotDataItem(
            pen=pg.mkPen('w', width=2)
        )
        self.plot.addItem(self.segment_item)

        # --------------------------------------------------------
        # DRAGGABLE PIVOT POINT
        # --------------------------------------------------------
        self.pivot_item = DraggablePoint(self.state, vb)
        self.plot.addItem(self.pivot_item)

        # --------------------------------------------------------
        # SECOND POINT (P2) — NOT DRAGGABLE
        # --------------------------------------------------------
        self.p2_item = pg.ScatterPlotItem(
            symbol='s',
            size=10,
            brush='w',
            pen=pg.mkPen(0, 0, 0, width=1)
        )
        self.plot.addItem(self.p2_item)

        # --------------------------------------------------------
        # PROJECTIVE RAY LINE
        # --------------------------------------------------------
        self.ray_item = pg.PlotDataItem(
            pen=pg.mkPen('y', width=2)
        )
        self.plot.addItem(self.ray_item)

        # --------------------------------------------------------
        # ANGLE ARC ITEM
        # --------------------------------------------------------
        self.angle_arc_item = pg.PlotDataItem(
            pen=pg.mkPen('y', width=2)
        )
        self.plot.addItem(self.angle_arc_item)

        # --------------------------------------------------------
        # CONTROLS FOR U, u, r2, theta2
        # --------------------------------------------------------
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setContentsMargins(0, 6, 0, 0)

        # U spinbox
        self.spinU = QtWidgets.QDoubleSpinBox()
        self.spinU.setDecimals(3)
        self.spinU.setRange(0.001, 1e6)
        self.spinU.setSingleStep(0.5)
        self.spinU.setValue(self.state.U)
        self.spinU.setPrefix("U = ")

        # u spinbox
        self.spinu = QtWidgets.QDoubleSpinBox()
        self.spinu.setDecimals(3)
        self.spinu.setRange(0.0, self.state.U)
        self.spinu.setSingleStep(0.1)
        self.spinu.setValue(self.state.u)
        self.spinu.setPrefix("u = ")

        # r2 spinbox
        self.spinr2 = QtWidgets.QDoubleSpinBox()
        self.spinr2.setDecimals(3)
        self.spinr2.setRange(0.0, 100.0)
        self.spinr2.setValue(self.state.r2)
        self.spinr2.setSingleStep(0.1)
        self.spinr2.setPrefix("r₂ = ")

        # theta2 spinbox
        self.spintheta2 = QtWidgets.QDoubleSpinBox()
        self.spintheta2.setDecimals(3)
        self.spintheta2.setRange(-1000.0, 1000.0)
        self.spintheta2.setValue(self.state.theta2)
        self.spintheta2.setSingleStep(0.05)
        self.spintheta2.setPrefix("θ₂ = ")

        control_layout.addWidget(self.spinU)
        control_layout.addWidget(self.spinu)
        control_layout.addWidget(self.spinr2)
        control_layout.addWidget(self.spintheta2)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

        # --------------------------------------------------------
        # SIGNAL CONNECTIONS
        # --------------------------------------------------------
        self.spinU.valueChanged.connect(self._on_spinU)
        self.spinu.valueChanged.connect(self._on_spinu)
        self.spinr2.valueChanged.connect(self._on_spinr2)
        self.spintheta2.valueChanged.connect(self._on_spintheta2)
        self.state.changed.connect(self._update)

        # Initial update
        self._update()

    # ============================================================
    # CONTROL HANDLERS
    # ============================================================

    def _on_spinU(self, val):      self.state.set_U(val)
    def _on_spinu(self, val):      self.state.set_u_from_line(val)
    def _on_spinr2(self, val):     self.state.set_r2(val)
    def _on_spintheta2(self, val): self.state.set_theta2(val)

    # ============================================================
    # MAIN UPDATE FUNCTION
    # ============================================================

    def _update(self):
        if self._updating:
            return

        self._updating = True

        # --------------------------------------------------------
        # Base values
        # --------------------------------------------------------
        U = self.state.U
        u = self.state.u
        theta2 = self.state.theta2
        r2 = self.state.r2

        # --------------------------------------------------------
        # Update segment [0, U]
        # --------------------------------------------------------
        self.segment_item.setData([0.0, U], [0.0, 0.0])

        # --------------------------------------------------------
        # Pivot point P1 = (u, 0)
        # --------------------------------------------------------
        self.pivot_item.setData([u], [0.0])

        # --------------------------------------------------------
        # Compute second point P2
        # --------------------------------------------------------
        x2 = u + r2 * math.cos(theta2)
        y2 = 0.0 + r2 * math.sin(theta2)
        self.p2_item.setData([x2], [y2])

        # --------------------------------------------------------
        # Draw ray from pivot to P2
        # --------------------------------------------------------
        self.ray_item.setData([u, x2], [0.0, y2])

        # --------------------------------------------------------
        # Draw angle arc (centered at pivot)
        # --------------------------------------------------------
        arc_x = []
        arc_y = []
        radius = max(0.25 * r2, 0.25)

        # sample arc
        samples = 64
        for t in np.linspace(0, theta2, samples):
            arc_x.append(u + radius * math.cos(t))
            arc_y.append(0.0 + radius * math.sin(t))

        self.angle_arc_item.setData(arc_x, arc_y)

        # --------------------------------------------------------
        # View range
        # --------------------------------------------------------
        margin = max(1.0, U * 0.1, abs(x2) * 0.2, abs(y2) * 0.2)
        self.plot.setXRange(-margin, U + margin, padding=0.0)
        self.plot.setYRange(-margin, margin, padding=0.0)

        # --------------------------------------------------------
        # Update spinboxes (avoid loops)
        # --------------------------------------------------------
        self.spinU.blockSignals(True)
        self.spinU.setValue(U)
        self.spinU.blockSignals(False)

        self.spinu.blockSignals(True)
        self.spinu.setRange(0.0, U)
        self.spinu.setValue(u)
        self.spinu.blockSignals(False)

        self.spinr2.blockSignals(True)
        self.spinr2.setValue(r2)
        self.spinr2.blockSignals(False)

        self.spintheta2.blockSignals(True)
        self.spintheta2.setValue(theta2)
        self.spintheta2.blockSignals(False)

        # --------------------------------------------------------
        # Address output
        # --------------------------------------------------------

        # Address output
        text = f"mode=2D | U={U:.3f}, u={u:.3f} | r₂={r2:.3f}, θ₂={theta2:.3f}"
        if hasattr(CodexDiagramState, "write_address"):
            CodexDiagramState.write_address("2d", text)


        self._updating = False

# ============================================================
# MODULE 3 — Diagram3DWidget
# ============================================================
# 3D interactive diagram with:
#   • Square base plane
#   • Pivot point P1 (sx, sy)
#   • Projective ray from P1 at angle theta2
#   • Second point P2 = (sx + r2*cosθ2, sy + r2*sinθ2, z2)
#   • User-defined parametric path x(t), y(t), z(t)
#   • User-defined sub-spaces (visual overlays)
# ============================================================



class Diagram3DWidget(QtWidgets.QWidget):

    def __init__(self, state: ModelState, parent=None):
        super().__init__(parent)

        self.state = state
        self._updating = False

        # Scaling factor: normalised square → world space
        self.RADIUS = 1.6

        # List of GL items for subspaces
        self.subspace_items = []
        self.subspace_defs = []

        # =====================================================
        # MAIN LAYOUT
        # =====================================================
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # =====================================================
        # 3D VIEW
        # =====================================================
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("black")
        self.view.setCameraPosition(distance=6.0, elevation=35, azimuth=45)
        layout.addWidget(self.view, 1)

        # -----------------------------------------------------
        # Base grid
        # -----------------------------------------------------
        grid = gl.GLGridItem()
        grid.setSize(x=4, y=4)
        grid.setSpacing(x=0.5, y=0.5)
        grid.setColor((0.3, 0.3, 0.3, 1.0))
        self.view.addItem(grid)

        # -----------------------------------------------------
        # Square boundary
        # -----------------------------------------------------
        R = self.RADIUS
        square_vertices = np.array([
            [-R, -R, 0],
            [ R, -R, 0],
            [ R,  R, 0],
            [-R,  R, 0],
            [-R, -R, 0]
        ])
        self.square_item = gl.GLLinePlotItem(
            pos=square_vertices,
            color=(1,1,1,1),
            width=2,
            mode='line_strip'
        )
        self.view.addItem(self.square_item)

        # -----------------------------------------------------
        # Pivot point P1 in 3D (small sphere)
        # -----------------------------------------------------
        meshP1 = gl.MeshData.sphere(rows=16, cols=32, radius=0.08)
        self.pivot_mesh = gl.GLMeshItem(
            meshdata=meshP1,
            smooth=True,
            color=(1,1,1,1),
            shader='shaded'
        )
        self.view.addItem(self.pivot_mesh)

        # -----------------------------------------------------
        # Second point P2 (square point)
        # -----------------------------------------------------
        meshP2 = gl.MeshData.sphere(rows=16, cols=32, radius=0.08)
        self.second_mesh = gl.GLMeshItem(
            meshdata=meshP2,
            smooth=True,
            color=(1,1,0,1),
            shader='shaded'
        )
        self.view.addItem(self.second_mesh)

        # -----------------------------------------------------
        # Projective ray line (P1 → P2)
        # -----------------------------------------------------
        self.ray_gl = gl.GLLinePlotItem(
            pos=np.zeros((2,3)),
            color=(1,1,0,1),
            width=3,
            mode='lines'
        )
        self.view.addItem(self.ray_gl)

        # -----------------------------------------------------
        # User-defined path GL line
        # -----------------------------------------------------
        self.path_gl = gl.GLLinePlotItem(
            pos=np.zeros((10,3)),
            color=(0.8,0.8,0.8,1),
            width=2,
            mode='line_strip'
        )
        self.view.addItem(self.path_gl)

        # =====================================================
        # CONTROL PANEL
        # =====================================================
        control = QtWidgets.QHBoxLayout()
        control.setContentsMargins(0,6,0,0)

        # x̄ spinbox
        self.spin_sx = QtWidgets.QDoubleSpinBox()
        self.spin_sx.setDecimals(3)
        self.spin_sx.setRange(-1.0, 1.0)
        self.spin_sx.setSingleStep(0.02)
        self.spin_sx.setPrefix("x̄ = ")
        control.addWidget(self.spin_sx)

        # ȳ spinbox
        self.spin_sy = QtWidgets.QDoubleSpinBox()
        self.spin_sy.setDecimals(3)
        self.spin_sy.setRange(-1.0, 1.0)
        self.spin_sy.setSingleStep(0.02)
        self.spin_sy.setPrefix("ȳ = ")
        control.addWidget(self.spin_sy)

        # r₂ spinbox
        self.spin_r2 = QtWidgets.QDoubleSpinBox()
        self.spin_r2.setDecimals(3)
        self.spin_r2.setRange(0.0, 100.0)
        self.spin_r2.setSingleStep(0.1)
        self.spin_r2.setPrefix("r₂ = ")
        control.addWidget(self.spin_r2)

        # θ₂ spinbox
        self.spin_theta2 = QtWidgets.QDoubleSpinBox()
        self.spin_theta2.setDecimals(3)
        self.spin_theta2.setRange(-1000,1000)
        self.spin_theta2.setSingleStep(0.05)
        self.spin_theta2.setPrefix("θ₂ = ")
        control.addWidget(self.spin_theta2)

        # z₂ spinbox (3D elevation)
        self.spin_z2 = QtWidgets.QDoubleSpinBox()
        self.spin_z2.setDecimals(3)
        self.spin_z2.setRange(-10,10)
        self.spin_z2.setSingleStep(0.05)
        self.spin_z2.setPrefix("z₂ = ")
        control.addWidget(self.spin_z2)

        control.addStretch(1)
        layout.addLayout(control)

        # =====================================================
        # SUB-SPACE DEFINITIONS
        # =====================================================
        sub_box = QtWidgets.QGroupBox("Sub-spaces (expr per line)")
        sub_layout = QtWidgets.QVBoxLayout(sub_box)

        self.sub_edit = QtWidgets.QPlainTextEdit()
        self.sub_edit.setPlaceholderText(
            "upperRight: sx>=0 and sy>=0\n"
            "diamond: abs(sx)+abs(sy) < 0.7\n"
            "ring: 0.2 < math.hypot(sx,sy) < 0.7"
        )
        self.sub_edit.setFixedHeight(90)
        self.sub_edit.setStyleSheet(
            "background-color:black; color:white; border:1px solid #444;"
        )
        sub_layout.addWidget(self.sub_edit)

        self.btn_apply_sub = QtWidgets.QPushButton("Apply subspaces")
        sub_layout.addWidget(self.btn_apply_sub)

        layout.addWidget(sub_box)

        # =====================================================
        # USER-DEFINED PATH
        # =====================================================
        path_box = QtWidgets.QGroupBox("Path x(t), y(t), z(t)   0 ≤ t ≤ 1")
        path_layout = QtWidgets.QHBoxLayout(path_box)

        form = QtWidgets.QFormLayout()

        self.expr_x = QtWidgets.QLineEdit("sx1 + t*(sx2 - sx1)")
        self.expr_y = QtWidgets.QLineEdit("sy1 + t*(sy2 - sy1)")
        self.expr_z = QtWidgets.QLineEdit("0.4 * math.sin(3*t)")

        form.addRow("x(t) =", self.expr_x)
        form.addRow("y(t) =", self.expr_y)
        form.addRow("z(t) =", self.expr_z)

        path_layout.addLayout(form)

        self.btn_apply_path = QtWidgets.QPushButton("Build path")
        path_layout.addWidget(self.btn_apply_path)

        layout.addWidget(path_box)

        # =====================================================
        # SIGNAL CONNECTIONS
        # =====================================================
        self.state.changed.connect(self._update)

        self.spin_sx.valueChanged.connect(self._on_sx)
        self.spin_sy.valueChanged.connect(self._on_sy)
        self.spin_r2.valueChanged.connect(self._on_r2)
        self.spin_theta2.valueChanged.connect(self._on_theta2)
        self.spin_z2.valueChanged.connect(self._on_z2)

        self.btn_apply_sub.clicked.connect(self._on_apply_subspaces)
        self.btn_apply_path.clicked.connect(self._build_path)

        # Initialize
        self._update()

    # ============================================================
    # CONTROL HANDLERS
    # ============================================================

    def _on_sx(self, val):      self.state.set_from_square(val, self.state.sy)
    def _on_sy(self, val):      self.state.set_from_square(self.state.sx, val)
    def _on_r2(self, val):      self.state.set_r2(val)
    def _on_theta2(self, val):  self.state.set_theta2(val)
    def _on_z2(self, val):      self.state.set_z2(val)

    # ============================================================
    # SUBSPACES
    # ============================================================

    def _on_apply_subspaces(self):
        text = self.sub_edit.toPlainText()
        lines = text.splitlines()

        # Remove old
        for it in self.subspace_items:
            try: self.view.removeItem(it)
            except: pass
        self.subspace_items.clear()
        self.subspace_defs.clear()

        idx = 0
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if ":" in s:
                name, expr = map(str.strip, s.split(":", 1))
                if not name:
                    name = f"sub{idx}"
            else:
                name = f"sub{idx}"
                expr = s

            self.subspace_defs.append((name, expr))
            self._build_subspace_overlay(expr, idx)
            idx += 1

        self._update()

    def _build_subspace_overlay(self, expr: str, idx: int):
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "abs": abs, "min": min, "max": max
        }

        pts = []
        R = self.RADIUS

        N = 27
        for i in range(N):
            sx = -1 + 2*i/(N-1)
            for j in range(N):
                sy = -1 + 2*j/(N-1)

                env = dict(
                    sx=sx, sy=sy, x=sx, y=sy,
                    r=math.hypot(sx,sy),
                    theta=math.atan2(sy,sx),
                    U=self.state.U,
                    u=self.state.u,
                )

                try:
                    if bool(eval(expr, safe_globals, env)):
                        pts.append([sx*R, sy*R, 0])
                except:
                    pass

        if not pts:
            return

        color = (0.4 + 0.15*(idx%3),)*3 + (0.8,)
        item = gl.GLScatterPlotItem(
            pos=np.array(pts,float),
            color=color,
            size=4
        )
        self.view.addItem(item)
        self.subspace_items.append(item)

    # ============================================================
    # PATH BUILDING
    # ============================================================

    def _build_path(self):
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "abs": abs, "min": min, "max": max
        }

        # pivot P1
        sx1, sy1 = self.state.sx, self.state.sy
        # second point P2
        sx2, sy2, z2 = self.state.get_second_point_3d_square()

        ex = self.expr_x.text().strip()
        ey = self.expr_y.text().strip()
        ez = self.expr_z.text().strip()

        pts = []
        for t in np.linspace(0,1,300):
            env = dict(
                t=t,
                sx1=sx1, sy1=sy1,
                sx2=sx2, sy2=sy2,
                z2=z2
            )
            try:
                x = eval(ex, safe_globals, env)
                y = eval(ey, safe_globals, env)
                z = eval(ez, safe_globals, env)
            except:
                continue

            pts.append([x*self.RADIUS, y*self.RADIUS, z])

        if pts:
            self.path_gl.setData(pos=np.array(pts,float))

    # ============================================================
    # MAIN UPDATE FUNCTION
    # ============================================================

    def _update(self):
        if self._updating:
            return
        self._updating = True

        sx = self.state.sx
        sy = self.state.sy
        r2 = self.state.r2
        theta2 = self.state.theta2
        z2 = self.state.z2

        # -----------------------------------------------------
        # Pivot P1
        # -----------------------------------------------------
        x1 = self.RADIUS * sx
        y1 = self.RADIUS * sy
        z1 = 0

        self.pivot_mesh.resetTransform()
        self.pivot_mesh.translate(x1, y1, z1)

        # -----------------------------------------------------
        # Second P2
        # -----------------------------------------------------
        sx2 = sx + r2*math.cos(theta2)
        sy2 = sy + r2*math.sin(theta2)

        x2 = self.RADIUS * sx2
        y2 = self.RADIUS * sy2

        self.second_mesh.resetTransform()
        self.second_mesh.translate(x2, y2, z2)

        # -----------------------------------------------------
        # Ray P1 → P2
        # -----------------------------------------------------
        self.ray_gl.setData(
            pos=np.array([[x1,y1,0],[x2,y2,z2]], float)
        )

        # -----------------------------------------------------
        # Update path (if previously built)
        # -----------------------------------------------------
        self._build_path()

        # -----------------------------------------------------
        # Update controls (block signals)
        # -----------------------------------------------------
        self.spin_sx.blockSignals(True);    self.spin_sx.setValue(sx);     self.spin_sx.blockSignals(False)
        self.spin_sy.blockSignals(True);    self.spin_sy.setValue(sy);     self.spin_sy.blockSignals(False)
        self.spin_r2.blockSignals(True);    self.spin_r2.setValue(r2);     self.spin_r2.blockSignals(False)
        self.spin_theta2.blockSignals(True);self.spin_theta2.setValue(theta2);self.spin_theta2.blockSignals(False)
        self.spin_z2.blockSignals(True);    self.spin_z2.setValue(z2);     self.spin_z2.blockSignals(False)

        # -----------------------------------------------------
        # Address line
        # -----------------------------------------------------
        text = (
            f"mode=3D | x̄={sx:.3f},ȳ={sy:.3f} | "
            f"r₂={r2:.3f}, θ₂={theta2:.3f} | z₂={z2:.3f}"
        )
        if hasattr(CodexDiagramState, "write_address"):
            CodexDiagramState.write_address("3d", text)

        self._updating = False

# ============================================================
# MODULE 4 — CodexDiagramState Helper
# ============================================================
# This helper manages the "address" text shown at the bottom
# of the Codex main window. It remembers the most recent 2D
# and 3D address content and shows the correct one depending
# on which mode is active.
# ============================================================

#from PyQt6 import QtWidgets


class _CodexDiagramStateHelper:
    """
    Tracks:
      - The currently active mode: "2d" or "3d"
      - The last displayed address text for each mode
      - Writes to the shared QLabel in MainWindow
    """

    def __init__(self, label: QtWidgets.QLabel):
        self._label = label
        self.active_mode = "2d"
        self.last = {"2d": "", "3d": ""}

    # --------------------------------------------------------
    # Called by MainWindow when user switches tabs
    # --------------------------------------------------------
    def set_active_mode(self, mode: str):
        self.active_mode = mode
        current_text = self.last.get(mode, "")
        self._label.setText(current_text)

    # --------------------------------------------------------
    # Called by widgets when they recompute address text
    # --------------------------------------------------------
    def write_address(self, mode: str, text: str):
        self.last[mode] = text
        if mode == self.active_mode:
            self._label.setText(text)


# Global shared instance created by MainWindow
CodexDiagramState = None

# ============================================================
# MODULE 5 — MainWindow
# ============================================================
# The top-level application window:
#   - Instantiates ModelState
#   - Contains Diagram2DWidget + Diagram3DWidget
#   - Manages tab switching
#   - Controls the address bar
#   - Wires global CodexDiagramState
# ============================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------
        # Window properties
        # --------------------------------------------------------
        self.setWindowTitle("Codex Diagram — 2D / 3D Projective Model")
        self.resize(1100, 800)

        # --------------------------------------------------------
        # Create shared state
        # --------------------------------------------------------
        self.state = ModelState(U=12.0, parent=self)

        # --------------------------------------------------------
        # CENTRAL WIDGET AND LAYOUT
        # --------------------------------------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ========================================================
        # HEADER: Title + Tabs + Address Toggle
        # ========================================================
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0,0,0,0)
        main_layout.addLayout(header)

        # Title
        title_lbl = QtWidgets.QLabel("Diagram")
        title_lbl.setStyleSheet("color:#CCCCCC; font-weight:500; padding:2px;")
        header.addWidget(title_lbl)
        header.addSpacing(12)

        # Tabs
        self.btn_tab2d = QtWidgets.QPushButton("2D")
        self.btn_tab3d = QtWidgets.QPushButton("3D")
        for btn in (self.btn_tab2d, self.btn_tab3d):
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton {"
                "  background-color: #000;"
                "  color: #BBB;"
                "  border: 1px solid #444;"
                "  border-radius: 11px;"
                "  padding: 4px 10px;"
                "}"
                "QPushButton:checked {"
                "  background-color: #FFF;"
                "  color: #000;"
                "}"
            )

        self.btn_tab2d.setChecked(True)

        tabs = QtWidgets.QHBoxLayout()
        tabs.setSpacing(4)
        tabs.addWidget(self.btn_tab2d)
        tabs.addWidget(self.btn_tab3d)
        header.addLayout(tabs)

        header.addStretch(1)

        # Address toggle
        self.btn_addr_toggle = QtWidgets.QPushButton("addr")
        self.btn_addr_toggle.setCheckable(True)
        self.btn_addr_toggle.setChecked(False)
        self.btn_addr_toggle.setStyleSheet(
            "QPushButton {"
            "  background-color: #000;"
            "  color: #DDD;"
            "  border: 1px solid #555;"
            "  border-radius: 11px;"
            "  padding: 4px 10px;"
            "}"
            "QPushButton:checked {"
            "  background-color: #FFF;"
            "  color: #000;"
            "}"
        )
        header.addWidget(self.btn_addr_toggle)

        # ========================================================
        # STACKED VIEW (2D / 3D)
        # ========================================================
        self.stack = QtWidgets.QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        # 2D view
        self.widget2d = Diagram2DWidget(self.state)
        self.stack.addWidget(self.widget2d)

        # 3D view
        self.widget3d = Diagram3DWidget(self.state)
        self.stack.addWidget(self.widget3d)

        # ========================================================
        # ADDRESS BAR
        # ========================================================
        self.addr_row = QtWidgets.QWidget()
        addr_layout = QtWidgets.QHBoxLayout(self.addr_row)
        addr_layout.setContentsMargins(0,0,0,0)

        self.addr_label = QtWidgets.QLabel("")
        self.addr_label.setStyleSheet(
            "QLabel {"
            "  background-color:#000;"
            "  color:#EEE;"
            "  border:1px solid #444;"
            "  border-radius:11px;"
            "  padding:4px 10px;"
            "}"
        )
        addr_layout.addWidget(self.addr_label)
        addr_layout.addStretch(1)

        main_layout.addWidget(self.addr_row)
        self.addr_row.setVisible(False)

        # --------------------------------------------------------
        # GLOBAL CodexDiagramState setup
        # --------------------------------------------------------
        global CodexDiagramState
        CodexDiagramState = _CodexDiagramStateHelper(self.addr_label)

        # ========================================================
        # SIGNALS
        # ========================================================
        self.btn_tab2d.clicked.connect(self._switch_to_2d)
        self.btn_tab3d.clicked.connect(self._switch_to_3d)
        self.btn_addr_toggle.toggled.connect(self._toggle_address)
        # STATUS BAR
        self.status = self.statusBar()
        self.status.showMessage("Ready.")
        
        self.state.changed.connect(self._update_status)
        self.state.changed.emit()



    # ============================================================
    # TAB SWITCHING
    # ============================================================

    def _switch_to_2d(self, checked: bool):
        if checked:
            self.btn_tab3d.setChecked(False)
            self.stack.setCurrentWidget(self.widget2d)
            CodexDiagramState.set_active_mode("2d")

    def _switch_to_3d(self, checked: bool):
        if checked:
            self.btn_tab2d.setChecked(False)
            self.stack.setCurrentWidget(self.widget3d)
            CodexDiagramState.set_active_mode("3d")

    # ============================================================
    # ADDRESS BAR TOGGLE
    # ============================================================

    def _toggle_address(self, on: bool):
        self.addr_row.setVisible(on)

    # ============================================================
    # STATUS BAR UPDATE
    # ============================================================

    def _update_status(self):
        s = self.state
        msg = (
            f"U={s.U:.3f} | u={s.u:.3f} | "
            f"x̄={s.sx:.3f} | ȳ={s.sy:.3f} | "
            f"r₂={s.r2:.3f} | θ₂={s.theta2:.3f} | z₂={s.z2:.3f}"
        )
        self.status.showMessage(msg)

# ============================================================
# MODULE 6 — Entry Point
# ============================================================

# If single-file: (the final combined Codex.py will override this import)
# from .main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
