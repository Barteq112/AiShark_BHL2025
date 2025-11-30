import React, { useEffect, useRef, useState } from 'react';
import { 
  select, 
  geoMercator, 
  geoPath, 
  pie, 
  arc, 
  PieArcDatum,
  zoom,
} from 'd3';
import { CountryNode, MapTooltip } from '../types';
import { EUROPE_GEOJSON_URL } from '../constants';

interface EuropeMapProps {
  countries: CountryNode[]; // <--- Dodano: dane z backendu przekazane przez Dashboard
  onCountrySelect: (node: CountryNode) => void;
  selectedCountry: CountryNode | null;
}

const EuropeMap: React.FC<EuropeMapProps> = ({ countries, onCountrySelect, selectedCountry }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [geoData, setGeoData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [tooltip, setTooltip] = useState<MapTooltip | null>(null);

  // 1. Pobieranie GeoJSON (kontury mapy)
  useEffect(() => {
    fetch(EUROPE_GEOJSON_URL)
      .then((response) => response.json())
      .then((data) => {
        setGeoData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load map data', err);
        setLoading(false);
      });
  }, []);

  // 2. Rysowanie mapy i nanoszenie danych z props (countries)
  useEffect(() => {
    // Rysujemy tylko wtedy, gdy mamy już mapę (geoData) i dane z backendu (countries)
    if (!geoData || !svgRef.current || countries.length === 0) return;

    const svg = select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Clear previous renders
    svg.selectAll('*').remove();

    // Setup Zoom Behavior
    const g = svg.append('g');

    const zoomBehavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 8])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        g.selectAll('path.country-shape').attr('stroke-width', 0.5 / event.transform.k);
      });

    svg.call(zoomBehavior)
       .style('cursor', 'move');

    // Setup Projection
    const projection = geoMercator()
      .center([10, 50])
      .scale(width * 0.9) 
      .translate([width / 2, height / 2]);

    const pathGenerator = geoPath().projection(projection);

    // Draw Map Background
    g.selectAll('path.country-shape')
      .data(geoData.features)
      .enter()
      .append('path')
      .attr('class', 'country-shape')
      .attr('d', pathGenerator as any)
      .attr('fill', '#1e293b')
      .attr('stroke', '#334155')
      .attr('stroke-width', 0.5)
      .on('mouseover', function () {
        select(this).attr('fill', '#0f172a');
      })
      .on('mouseout', function () {
        select(this).attr('fill', '#1e293b');
      });

    // Draw Country Nodes from Props
    const nodeCoords: Record<string, [number, number]> = {};
    
    // Używamy props.countries zamiast stałej COUNTRIES
    countries.forEach(c => {
        const coords = projection([c.lon, c.lat]);
        if (coords) nodeCoords[c.id] = coords;
    });

    countries.forEach((country) => {
      const coords = nodeCoords[country.id];
      if (!coords) return;
      const [x, y] = coords;

      const group = g.append('g')
        .attr('transform', `translate(${x}, ${y})`)
        .style('cursor', 'pointer')
        .on('click', (event) => {
            event.stopPropagation();
            onCountrySelect(country);
        })
        .on('mouseenter', (event) => {
            setTooltip({
                x: event.clientX,
                y: event.clientY,
                data: country
            });
        })
        .on('mouseleave', () => {
            setTooltip(null);
        });

      // Pie Data
      const pieData = [
          { type: 'renewable', value: country.renewablePercentage },
          { type: 'non-renewable', value: 100 - country.renewablePercentage }
      ];

      const radius = 14;
      const pieGenerator = pie<{type: string, value: number}>().value(d => d.value).sort(null);
      const arcGenerator = arc<PieArcDatum<{type: string, value: number}>>()
          .innerRadius(radius - 3)
          .outerRadius(radius);

      // Draw Donut segments
      group.selectAll('path.donut-segment')
          .data(pieGenerator(pieData))
          .enter()
          .append('path')
          .attr('class', 'donut-segment')
          .attr('d', arcGenerator)
          .attr('fill', d => d.data.type === 'renewable' ? '#10b981' : '#475569')
          .attr('opacity', 0.9);

      // Center Dot
      group.append('circle')
          .attr('r', 5)
          .attr('fill', selectedCountry?.id === country.id ? '#fff' : '#34d399')
          .attr('stroke', '#020617')
          .attr('stroke-width', 1);
      
      // Pulse effect for selected
      if (selectedCountry?.id === country.id) {
          group.append('circle')
            .attr('r', radius + 5)
            .attr('fill', 'none')
            .attr('stroke', '#34d399')
            .attr('stroke-width', 1)
            .attr('opacity', 0.5)
            .classed('animate-ping', true);
      }
    });

  // Dodaliśmy 'countries' do zależności, żeby mapa przerysowała się po przyjściu danych z API
  }, [geoData, selectedCountry, onCountrySelect, countries]);

  if (loading) {
      return (
          <div className="w-full h-full flex items-center justify-center bg-slate-900 text-emerald-500 animate-pulse">
              Loading Geospatial Data...
          </div>
      );
  }

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden rounded-xl border border-slate-800 shadow-2xl">
      <svg ref={svgRef} className="w-full h-full" />
      
      <div className="absolute top-4 right-4 flex flex-col gap-2">
           <div className="bg-slate-900/80 backdrop-blur text-slate-400 p-2 rounded text-[10px] border border-slate-700">
              Scroll to Zoom • Drag to Pan
           </div>
      </div>

      <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-sm p-3 rounded-lg border border-slate-700 text-xs pointer-events-none">
          <div className="flex items-center mb-1">
              <span className="w-3 h-3 rounded-full bg-emerald-500 mr-2"></span>
              <span className="text-slate-300">Renewable Energy</span>
          </div>
          <div className="flex items-center">
              <span className="w-3 h-3 rounded-full bg-slate-600 mr-2"></span>
              <span className="text-slate-300">Grid Mix</span>
          </div>
      </div>

      {tooltip && (
          <div 
            className="fixed pointer-events-none z-50 bg-slate-900 border border-emerald-500/50 p-3 rounded-lg shadow-xl text-sm w-48"
            style={{ left: tooltip.x + 15, top: tooltip.y - 15 }}
          >
              <h4 className="font-bold text-emerald-400 mb-1">{tooltip.data.name}</h4>
              <p className="text-slate-300 text-xs mb-2">Region</p>
              
              <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-slate-400">Green Energy:</span>
                  <span className={`font-mono font-bold ${tooltip.data.renewablePercentage > 50 ? 'text-emerald-400' : 'text-yellow-500'}`}>
                      {tooltip.data.renewablePercentage}%
                  </span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-emerald-500" 
                    style={{ width: `${tooltip.data.renewablePercentage}%` }}
                  ></div>
              </div>
          </div>
      )}
    </div>
  );
};

export default EuropeMap;