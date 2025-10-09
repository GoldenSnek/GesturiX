import { Stack } from 'expo-router';
import React from 'react';

const AuthStackLayout = () => {
  return (
    <Stack
      screenOptions={{
        headerShown: false, animation: 'slide_from_right'
      }}
    >
      <Stack.Screen name="LandingPage" />
      <Stack.Screen name="Login" />
      <Stack.Screen name="SignUp" />
    </Stack>
  );
};

export default AuthStackLayout;