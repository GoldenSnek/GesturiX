import Constants from 'expo-constants'; //

const FALLBACK_IP = '10.35.54.193'; 
export const SERVER_PORT = '8000';

const getDevServerIP = () => {
  const hostUri = Constants.expoConfig?.hostUri;
  
  if (hostUri) {
    const ip = hostUri.split(':')[0];
    return ip;
  }
  
  return FALLBACK_IP;
};

export const SERVER_IP = getDevServerIP();

export const API_BASE_URL = `http://${SERVER_IP}:${SERVER_PORT}`;

export const ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
  ENHANCE: `${API_BASE_URL}/enhance`,
};