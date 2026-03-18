import { use } from 'echarts/core'
import {
    BarChart,
    EffectScatterChart,
    HeatmapChart,
    LineChart,
    PieChart,
    RadarChart,
    ScatterChart,
} from 'echarts/charts'
import {
    DataZoomComponent,
    GeoComponent,
    GridComponent,
    LegendComponent,
    RadarComponent,
    TitleComponent,
    TooltipComponent,
    VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import * as echarts from 'echarts/core'

use([
    LineChart,
    PieChart,
    BarChart,
    RadarChart,
    ScatterChart,
    EffectScatterChart,
    HeatmapChart,
    TooltipComponent,
    TitleComponent,
    LegendComponent,
    GridComponent,
    RadarComponent,
    DataZoomComponent,
    GeoComponent,
    VisualMapComponent,
    CanvasRenderer,
])

export { echarts }
