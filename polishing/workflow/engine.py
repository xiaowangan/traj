"""Application facade for independently migrated numerical services (no Qt/OCC)."""
from polishing.io.zygo_reader import ZygoReaderMixin
from polishing.io.grid_reader import GridReaderMixin
from polishing.core.metrics import MetricsMixin
from polishing.core.simulation import SimulationMixin
from polishing.core.measurement_processing import ProcessingMixin
from polishing.core.transforms import TransformMixin
from polishing.core.surfaces import SurfaceMixin
from polishing.core.contact import ContactMixin
from polishing.core.removal import RemovalMixin
from polishing.core.dwell import DwellSolverMixin
from polishing.core.feed import DwellTrajectoryMixin
from polishing.io.path_export import MachinePathMixin


class DwellTimeEngine(ZygoReaderMixin, GridReaderMixin, MetricsMixin, SimulationMixin, ProcessingMixin, TransformMixin, SurfaceMixin, ContactMixin, RemovalMixin, DwellSolverMixin, DwellTrajectoryMixin, MachinePathMixin):
    """Compatibility API; implementation lives in the responsible service layers."""
