"""Report component registry."""

from ._funnel import FunnelComponent
from ._disease_breakdown import DiseaseBreakdownComponent
from ._spider import SpiderChartComponent
from ._heatmaps import HeatmapComponent
from ._bubble_heatmaps import BubbleHeatmapComponent
from ._loa import LOAComponent
from ._oncology import OncologyComponent
from ._timeline import TimelineComponent
from ._time_period import TimePeriodComponent
from ._modality_trend import ModalityTrendComponent
from ._sponsor import SponsorComponent
from ._candidate_summary import CandidateSummaryComponent

ALL_COMPONENTS: list[type] = [
    FunnelComponent,
    DiseaseBreakdownComponent,
    SpiderChartComponent,
    HeatmapComponent,
    BubbleHeatmapComponent,
    LOAComponent,
    OncologyComponent,
    TimelineComponent,
    TimePeriodComponent,
    ModalityTrendComponent,
    SponsorComponent,
    CandidateSummaryComponent,
]
