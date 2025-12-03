import Constants from 'expo-constants'; //

// Fallback IP (This is used if the app can't detect the bundler, e.g., in a production build)
// You can leave this as your last known working IP.
const FALLBACK_IP = '192.168.61.136'; 
export const SERVER_PORT = '8000';

const getDevServerIP = () => {
  // Constants.expoConfig?.hostUri contains the address of the machine running the bundler
  // Example value: "192.168.1.15:8081"
  const hostUri = Constants.expoConfig?.hostUri;
  
  if (hostUri) {
    // We split by ':' to remove the port and get just the IP address
    const ip = hostUri.split(':')[0];
    return ip;
  }
  
  // Return fallback if detection fails (e.g. standalone production builds)
  return FALLBACK_IP;
};

export const SERVER_IP = getDevServerIP();

export const API_BASE_URL = `http://${SERVER_IP}:${SERVER_PORT}`;

export const ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
  ENHANCE: `${API_BASE_URL}/enhance`,
};