export interface CountryNode {
  id: string;
  name: string;
  lat: number;
  lon: number;
  renewablePercentage: number; // 0-100
  facilities: string[];
  status: 'online' | 'maintenance' | 'offline';
}

export enum ComputeType {
  LIGHT = 'Light Computing',
  MEDIUM = 'Medium Computing',
  HEAVY = 'Heavy Computing',
}

export interface MapTooltip {
  x: number;
  y: number;
  data: CountryNode;
}
