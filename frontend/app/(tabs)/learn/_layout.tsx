// File: frontend/app/(tabs)/learn/_layout.tsx
import React from 'react';
import { View } from 'react-native';
import { Stack } from 'expo-router';
import { useTheme } from '../../../src/ThemeContext';

const LearnStack = () => {
  const { isDark } = useTheme();

  return (
    <View style={{ flex: 1, backgroundColor: isDark ? '#1A1A1A' : '#F8F8F8' }}>
      <Stack
        screenOptions={{
          headerShown: false,
          animation: 'fade',
          contentStyle: {
            backgroundColor: isDark ? '#1A1A1A' : '#F8F8F8',
          },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="letters" options={{ title: 'Letters' }} />
        <Stack.Screen name="numbers" options={{ title: 'Numbers' }} />
        <Stack.Screen name="phrases" options={{ title: 'Phrases' }} />
        <Stack.Screen name="saved" options={{ title: 'Saved Signs' }} />
      </Stack>
    </View>
  );
};

export default LearnStack;