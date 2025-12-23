import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function CorrelationHeatmap() {
  const svgRef = useRef();
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/data/feature_analysis.json')
      .then(res => res.json())
      .then(data => setData(data))
      .catch(err => console.error('Error loading feature analysis:', err));
  }, []);

  useEffect(() => {
    if (!data) return;

    const container = svgRef.current?.parentElement || svgRef.current;
    let resizeObserver;

    const draw = () => {
      // clear previous drawing + tooltips
      d3.select(svgRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-corr').remove();

      const correlations = data.correlations.slice(0, 15);
      const features = correlations.map(d => d.feature);
      const n = features.length;

      const containerWidth = container?.clientWidth || 800;
      const margin = { top: 80, right: 40, bottom: 120, left: 40 };
      const cellWidth = Math.min(120, Math.max(40, (containerWidth - margin.left - margin.right) / Math.max(1, n)));
      const cellHeight = 80;
      const width = Math.max(cellWidth * n, 300);
      const height = cellHeight;

      // create svg with viewBox for responsiveness
      const svg = d3.select(svgRef.current)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('width', '100%')
        .style('height', 'auto')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top - 30})`);

      const maxAbs = d3.max(correlations, d => Math.abs(d.correlation)) || 1;
      const colorScale = d3.scaleSequential().domain([0, maxAbs]).interpolator(d3.interpolateReds);

      const x = d3.scaleBand().domain(features).range([0, width]).padding(0.05);
      const y = d3.scaleBand().domain(['corr']).range([0, height]).padding(0.05);

      // tooltip (single global)
      const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'd3-tooltip d3-tooltip-corr')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0, 0, 0, 0.9)')
        .style('color', 'white')
        .style('padding', '8px 12px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('pointer-events', 'none')
        .style('z-index', '1000');

      // cells
      const cells = svg.selectAll('.cell')
        .data(correlations)
        .enter()
        .append('rect')
        .attr('class', 'cell')
        .attr('x', d => x(d.feature))
        .attr('y', () => y('corr'))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(Math.abs(d.correlation)))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('opacity', 0)
        .style('cursor', 'pointer');

      cells.transition().duration(600).delay((d, i) => i * 30).attr('opacity', 1);

      cells.on('mouseover', function(event, d) {
        d3.select(this).attr('stroke', '#333').attr('stroke-width', 3);
        tooltip.style('visibility', 'visible').html(`
          <strong>${d.feature.replace(/_/g, ' ')}</strong><br/>
          Correlation with Spam: <strong>${d.correlation.toFixed(4)}</strong><br/>
          <em>${d.correlation > 0 ? 'Positive' : 'Negative'} relationship</em>
        `);
      }).on('mousemove', function(event) {
        tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
      }).on('mouseout', function() {
        d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
        tooltip.style('visibility', 'hidden');
      });

      // values
      svg.selectAll('.correlation-text')
        .data(correlations)
        .enter()
        .append('text')
        .attr('class', 'correlation-text')
        .attr('x', d => x(d.feature) + x.bandwidth() / 2)
        .attr('y', d => y('corr') + y.bandwidth() / 2)
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('font-size', '11px')
        .attr('font-weight', '600')
        .attr('fill', d => Math.abs(d.correlation) > (maxAbs * 0.25) ? 'white' : '#333')
        .attr('opacity', 0)
        .text(d => d.correlation.toFixed(3))
        .transition()
        .duration(600)
        .delay((d, i) => i * 30 + 200)
        .attr('opacity', 1);

      // x axis
      svg.append('g')
        .attr('transform', `translate(0, ${height + 8})`)
        .call(d3.axisBottom(x).tickFormat(d => d.replace(/_/g, ' ')))
        .selectAll('text')
        .style('font-size', '11px')
        .style('fill', '#000')
        .style('text-anchor', 'end')
        .attr('transform', 'rotate(-45)')
        .attr('dx', '-0.6em')
        .attr('dy', '0.25em');

      // title
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', -40)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text('Correlation with Spam (Top features)');

      // legend
      const legendWidth = Math.min(360, width);
      const legendHeight = 12;
      const legendX = Math.max(0, (width - legendWidth) / 2);
      const legendY = height + Math.max(60, margin.bottom - 15);

      const defs = svg.append('defs');
      const gradient = defs.append('linearGradient').attr('id', 'correlation-gradient').attr('x1', '0%').attr('x2', '100%');
      gradient.selectAll('stop').data(d3.range(0, 1.01, 0.1)).enter().append('stop').attr('offset', d => `${d * 100}%`).attr('stop-color', t => colorScale(t * maxAbs));

      const legend = svg.append('g').attr('class', 'legend').attr('transform', `translate(${legendX}, ${legendY})`);
      legend.append('rect').attr('width', legendWidth).attr('height', legendHeight).style('fill', 'url(#correlation-gradient)').attr('stroke', '#ccc').attr('stroke-width', 1);

      const legendScale = d3.scaleLinear().domain([0, maxAbs]).range([0, legendWidth]);
      const legendAxis = d3.axisBottom(legendScale).ticks(5).tickFormat(d3.format('.2f'));
      legend.append('g')
        .attr('transform', `translate(0, ${legendHeight})`)
        .call(legendAxis)
        .selectAll('text')
        .attr('font-size', '11px')
        .attr('fill', '#000');

      legend.append('text').attr('x', legendWidth / 2).attr('y', -8).attr('text-anchor', 'middle').attr('font-size', '12px').attr('font-weight', '600').text('Absolute Correlation Strength');
    };

    draw();

    const onResize = () => draw();
    window.addEventListener('resize', onResize);

    if ('ResizeObserver' in window && container) {
      resizeObserver = new ResizeObserver(() => draw());
      resizeObserver.observe(container);
    }

    return () => {
      window.removeEventListener('resize', onResize);
      if (resizeObserver && container) resizeObserver.unobserve(container);
      d3.select(svgRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-corr').remove();
    };
  }, [data]);

  if (!data) {
    return <div className="loading">Loading correlations...</div>;
  }

  return (
    <div className="correlation-heatmap">
      <div className="chart-header">
        <h3>Feature Correlation with Spam</h3>
        <p className="chart-description">
          Top 15 features most correlated with spam messages
        </p>
      </div>
      <svg ref={svgRef}></svg>
    </div>
  );
}

export default CorrelationHeatmap;
