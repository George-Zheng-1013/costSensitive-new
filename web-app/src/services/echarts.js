import { use } from 'echarts/core'
import { BarChart, LineChart, PieChart, RadarChart } from 'echarts/charts'
import {
    GridComponent,
    LegendComponent,
    RadarComponent,
    TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import * as echarts from 'echarts/core'

use([
    LineChart,
    PieChart,
    BarChart,
    RadarChart,
    TooltipComponent,
    LegendComponent,
    GridComponent,
    RadarComponent,
    CanvasRenderer,
])

export { echarts }
