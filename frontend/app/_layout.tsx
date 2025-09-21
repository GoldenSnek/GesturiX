import { Stack } from 'expo-router';
import './global.css';
import React from 'react';

const RootLayout = () => {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
      }}
    >
      {/* This is the initial route, showing the landing page */}
      <Stack.Screen name="(stack)" /> 
      {/* This is a single screen for the tab navigator */}
      <Stack.Screen name="(tabs)" />
    </Stack>
  );
};

export default RootLayout;