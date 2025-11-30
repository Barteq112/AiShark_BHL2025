import { CountryNode } from './types';

export const COUNTRIES: CountryNode[] = [
  {
    id: 'no',
    name: 'Norway',
    lat: 60.47,
    lon: 8.46,
    renewablePercentage: 98,
    facilities: [
      'Lefdal Mine Datacenter',
      'Bulk Campus N01',
      'Green Mountain DC1',
      'Green Mountain DC2',
      'Green Mountain DC3'
    ],
    status: 'online'
  },
  {
    id: 'nl',
    name: 'Netherlands',
    lat: 52.13,
    lon: 5.29,
    renewablePercentage: 45,
    facilities: [
      'Google Eemshaven',
      'Equinix Amsterdam',
      'Interxion Amsterdam'
    ],
    status: 'online'
  },
  {
    id: 'de',
    name: 'Germany',
    lat: 51.16,
    lon: 10.45,
    renewablePercentage: 52,
    facilities: [
      'Equinix Frankfurt',
      'Hetzner Falkenstein',
      'Hetzner Nuremberg',
      'Interxion Frankfurt',
      'Global Switch Frankfurt',
      'Vantage Frankfurt',
      'Colt DCS Frankfurt'
    ],
    status: 'online'
  },
  {
    id: 'fr',
    name: 'France',
    lat: 46.22,
    lon: 2.21,
    renewablePercentage: 78,
    facilities: [
      'Data4 Paris-Saclay',
      'OVH Gravelines',
      'OVH Roubaix',
      'OVH Strasbourg',
      'Global Switch Paris',
      'Interxion Paris'
    ],
    status: 'maintenance'
  },
  {
    id: 'pl',
    name: 'Poland',
    lat: 51.91,
    lon: 19.14,
    renewablePercentage: 25,
    facilities: [
      'Atman',
      'Equinix Warsaw',
      'Beyond.pl',
      'Data4 Poland',
      '3S Data Center',
      'COIG / WASKO'
    ],
    status: 'online'
  },
  {
    id: 'it',
    name: 'Italy',
    lat: 41.87,
    lon: 12.56,
    renewablePercentage: 42,
    facilities: [
      'Aruba Global Cloud Data Center (IT3)',
      'Aruba IT1',
      'Aruba IT2',
      'SuperNAP Italia'
    ],
    status: 'online'
  },
];

// GeoJSON for Europe will be fetched dynamically, but we define the URL here.
// Using a reliable public source.
export const EUROPE_GEOJSON_URL = 'https://raw.githubusercontent.com/leakyMirror/map-of-europe/master/GeoJSON/europe.geojson';
