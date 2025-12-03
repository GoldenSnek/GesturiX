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
        <Stack.Screen name="videos" options={{ title: 'Video Lessons' }} />
        <Stack.Screen name="dictionary" options={{ title: 'Dictionary' }} />
        <Stack.Screen name="leaderboard" options={{ title: 'Leaderboard' }} />
      </Stack>
    </View>
  );
};

export default LearnStack;