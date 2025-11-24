// File: frontend/constants/ApiConfig.ts

// Kani lang usa ka IP Adress ilisi good to go na
export const SERVER_IP = '192.168.89.136'; 
export const SERVER_PORT = '8000';

export const API_BASE_URL = `http://${SERVER_IP}:${SERVER_PORT}`;

export const ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/predict`,
  ENHANCE: `${API_BASE_URL}/enhance`,
};