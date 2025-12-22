import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function DatasetOverview() {
  const pieRef = useRef();
  const sourceRef = useRef();
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('/data/feature_analysis.json')
      .then(res => res.json())
      .then(data => setData(data))
      .catch(err => console.error('Error loading feature analysis:', err));
  }, []);

  useEffect(() => {
    if (!data) return;

    const pieContainer = pieRef.current?.parentElement || pieRef.current;
    const sourceContainer = sourceRef.current?.parentElement || sourceRef.current;
    let pieRO, sourceRO;

    const drawPie = () => {
      d3.select(pieRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-pie').remove();

      const stats = data.overall_stats;
      const pieData = [
        { label: 'Spam', value: stats.spam_count, color: '#f44336' },
        { label: 'Ham', value: stats.ham_count, color: '#4CAF50' }
      ];

      const containerWidth = pieContainer?.clientWidth || 300;
      const size = Math.min(300, Math.max(200, containerWidth - 40));
      const width = size;
      const height = size + 40; // Add extra height to prevent overlap with title
      const radius = Math.min(width, height) / 2 - 24;

      const svg = d3.select(pieRef.current)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('width', '100%')
        .style('height', 'auto')
        .append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`); // Center vertically instead of height/4

      const pie = d3.pie().value(d => d.value).sort(null);
      const arc = d3.arc().innerRadius(radius * 0.4).outerRadius(radius);
      const arcHover = d3.arc().innerRadius(radius * 0.3).outerRadius(radius);

      const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'd3-tooltip d3-tooltip-pie')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0,0,0,0.85)')
        .style('color', '#fff')
        .style('padding', '8px 10px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('pointer-events', 'none')
        .style('z-index', '1000');

      const arcs = svg.selectAll('.arc')
        .data(pie(pieData))
        .enter()
        .append('g')
        .attr('class', 'arc');

      arcs.append('path')
        .attr('d', arc)
        .attr('fill', d => d.data.color)
        .attr('opacity', 0.95)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
          d3.select(this).transition().duration(180).attr('d', arcHover);
          tooltip.style('visibility', 'visible').html(`
            <strong>${d.data.label}</strong><br/>
            Count: <strong>${d.data.value.toLocaleString()}</strong><br/>
            Percentage: <strong>${(d.data.value / stats.total_messages * 100).toFixed(1)}%</strong>
          `);
        })
        .on('mousemove', function(event) {
          tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', function() {
          d3.select(this).transition().duration(180).attr('d', arc);
          tooltip.style('visibility', 'hidden');
        })
        .transition()
        .duration(700)
        .attrTween('d', function(d) {
          const i = d3.interpolate({ startAngle: 0, endAngle: 0 }, d);
          return t => arc(i(t));
        });

      arcs.append('text')
        .attr('transform', d => `translate(${arc.centroid(d)})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', 20)
        .attr('font-weight', '700')
        .attr('fill', '#fff')
        .attr('opacity', 0)
        .text(d => `${(d.data.value / stats.total_messages * 100).toFixed(1)}%`)
        .transition().delay(700).duration(400).attr('opacity', 1);

      // Center text showing total messages
      svg.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('font-size', 15)
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text(stats.total_messages.toLocaleString());

      // Legend positioned below the chart
      const legend = svg.append('g').attr('transform', `translate(${-width/2 + 20}, ${radius + 25})`);
      pieData.forEach((d, i) => {
        const g = legend.append('g').attr('transform', `translate(${i * (width / 2.5)}, 0)`);
        g.append('rect').attr('width', 20).attr('height', 20).attr('fill', d.color).attr('rx', 2);
        g.append('text')
          .attr('x', 24)
          .attr('y', 13)
          .attr('font-size', 13)
          .attr('font-weight', '600')
          .attr('fill', '#333')
          .text(`${d.label} (${d.value.toLocaleString()})`);
      });
    };

    const drawSource = () => {
      d3.select(sourceRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-source').remove();

      const sources = data.overall_stats.sources;
      const sourceData = Object.entries(sources).map(([key, value]) => ({ source: key, count: value }));

      const containerWidth = sourceContainer?.clientWidth || 420;
      const margin = { top: 20, right: 12, bottom: 60, left: 48 };
      const width = Math.max(280, containerWidth - 20) - margin.left - margin.right;
      const height = 300 - margin.top - margin.bottom;

      const svg = d3.select(sourceRef.current)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('width', '100%')
        .style('height', 'auto')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top + 20})`);

      const x = d3.scaleBand().domain(sourceData.map(d => d.source)).range([0, width]).padding(0.25);
      const y = d3.scaleLinear().domain([0, d3.max(sourceData, d => d.count) || 1]).nice().range([height, 0]);

      const colorScale = d3.scaleOrdinal().domain(sourceData.map(d => d.source)).range(d3.schemeTableau10);

      const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'd3-tooltip d3-tooltip-source')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background-color', 'rgba(0,0,0,0.85)')
        .style('color', '#fff')
        .style('padding', '8px 10px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('pointer-events', 'none')
        .style('z-index', '1000');

      svg.selectAll('.bar')
        .data(sourceData)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => x(d.source))
        .attr('width', x.bandwidth())
        .attr('y', height)
        .attr('height', 0)
        .attr('fill', d => colorScale(d.source))
        .attr('opacity', 0.9)
        .on('mouseover', function(event, d) {
          d3.select(this).attr('opacity', 1);
          tooltip.style('visibility', 'visible').html(`
            <strong>${d.source.toUpperCase()}</strong><br/>
            Messages: <strong>${d.count.toLocaleString()}</strong><br/>
            ${(d.count / data.overall_stats.total_messages * 100).toFixed(1)}% of total
          `);
        })
        .on('mousemove', function(event) {
          tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', function() {
          d3.select(this).attr('opacity', 0.9);
          tooltip.style('visibility', 'hidden');
        })
        .transition().duration(700).delay((d,i)=> i*80)
        .attr('y', d => y(d.count))
        .attr('height', d => height - y(d.count));

      svg.selectAll('.value-label')
        .data(sourceData)
        .enter()
        .append('text')
        .attr('class', 'value-label')
        .attr('x', d => x(d.source) + x.bandwidth() / 2)
        .attr('y', d => y(d.count) - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', '#333')
        .attr('opacity', 0)
        .text(d => d.count.toLocaleString())
        .transition().delay(700).duration(400).attr('opacity', 1);

      svg.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x)).selectAll('text').attr('font-size', 12);
      svg.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').attr('font-size', 11);
    };

    const drawAll = () => {
      drawPie();
      drawSource();
    };

    drawAll();
    const onResize = () => drawAll();
    window.addEventListener('resize', onResize);

    if ('ResizeObserver' in window && pieContainer) {
      pieRO = new ResizeObserver(() => drawPie());
      pieRO.observe(pieContainer);
    }
    if ('ResizeObserver' in window && sourceContainer) {
      sourceRO = new ResizeObserver(() => drawSource());
      sourceRO.observe(sourceContainer);
    }

    return () => {
      window.removeEventListener('resize', onResize);
      if (pieRO && pieContainer) pieRO.unobserve(pieContainer);
      if (sourceRO && sourceContainer) sourceRO.unobserve(sourceContainer);
      d3.select(pieRef.current).selectAll('*').remove();
      d3.select(sourceRef.current).selectAll('*').remove();
      d3.selectAll('.d3-tooltip-pie').remove();
      d3.selectAll('.d3-tooltip-source').remove();
    };
  }, [data]);

  if (!data) {
    return <div className="loading">Loading dataset overview...</div>;
  }

  return (
    <div className="dataset-overview">
      <h2 className="section-title">Dataset Overview</h2>
      <div className="overview-grid">
        <div className="overview-card">
          <h3>Spam vs Ham Distribution</h3>
          <svg ref={pieRef}></svg>
        </div>
        <div className="overview-card">
          <h3>Messages by Source</h3>
          <svg ref={sourceRef}></svg>
        </div>
      </div>
      <div className="stats-cards">
        <div className="stat-card">
          <div className="stat-value">{data.overall_stats.total_messages.toLocaleString()}</div>
          <div className="stat-label">Total Messages</div>
        </div>
        <div className="stat-card spam">
          <div className="stat-value">{data.overall_stats.spam_count.toLocaleString()}</div>
          <div className="stat-label">Spam Messages</div>
        </div>
        <div className="stat-card ham">
          <div className="stat-value">{data.overall_stats.ham_count.toLocaleString()}</div>
          <div className="stat-label">Ham Messages</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{Object.keys(data.overall_stats.sources).length}</div>
          <div className="stat-label">Data Sources</div>
        </div>
      </div>
    </div>
  );
}

export default DatasetOverview;
