import React from 'react';
import { Stack } from 'expo-router';
import { View, ActivityIndicator } from 'react-native';
import './global.css';
import { ThemeProvider } from '../src/ThemeContext';
import { AuthProvider } from '../src/AuthContext';
import { SettingsProvider } from '../src/SettingsContext'; // <--- Import

import { useFonts } from 'expo-font';

// 🧠 Inter
import {
  Inter_400Regular,
  Inter_700Bold,
  Inter_400Regular_Italic,
  Inter_700Bold_Italic,
} from '@expo-google-fonts/inter';

// 🧠 Montserrat
import {
  Montserrat_400Regular,
  Montserrat_600SemiBold,
  Montserrat_700Bold,
  Montserrat_400Regular_Italic,
  Montserrat_700Bold_Italic,
} from '@expo-google-fonts/montserrat';

// 🧠 Orbitron
import {
  Orbitron_400Regular,
  Orbitron_700Bold,
} from '@expo-google-fonts/orbitron';

// 🧠 Fredoka
import {
  Fredoka_400Regular,
  Fredoka_500Medium,
  Fredoka_600SemiBold,
  Fredoka_700Bold,
} from '@expo-google-fonts/fredoka';

// 🧠 Audiowide
import { Audiowide_400Regular } from '@expo-google-fonts/audiowide';

const RootLayout = () => {
  const [fontsLoaded] = useFonts({
    // Inter
    'Inter-Regular': Inter_400Regular,
    'Inter-Bold': Inter_700Bold,
    'Inter-Italic': Inter_400Regular_Italic,
    'Inter-BoldItalic': Inter_700Bold_Italic,

    // Montserrat
    'Montserrat-Regular': Montserrat_400Regular,
    'Montserrat-SemiBold': Montserrat_600SemiBold,
    'Montserrat-Bold': Montserrat_700Bold,
    'Montserrat-Italic': Montserrat_400Regular_Italic,
    'Montserrat-BoldItalic': Montserrat_700Bold_Italic,

    // Orbitron
    'Orbitron-Regular': Orbitron_400Regular,
    'Orbitron-Bold': Orbitron_700Bold,

    // Fredoka
    'Fredoka-Regular': Fredoka_400Regular,
    'Fredoka-Medium': Fredoka_500Medium,
    'Fredoka-SemiBold': Fredoka_600SemiBold,
    'Fredoka-Bold': Fredoka_700Bold,

    // Audiowide
    'Audiowide-Regular': Audiowide_400Regular,
  });

  if (!fontsLoaded) {
    return (
      <View className="flex-1 items-center justify-center bg-black">
        <ActivityIndicator color="#FF6B00" />
      </View>
    );
  }

  return (
    <ThemeProvider>
      <AuthProvider>
        {/* Wrap with SettingsProvider */}
        <SettingsProvider> 
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="(stack)" />
            <Stack.Screen name="(tabs)" />
          </Stack>
        </SettingsProvider>
      </AuthProvider>
    </ThemeProvider>
  );
};

export default RootLayout;